#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use std::fs::OpenOptions;
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use tauri::api::dialog;
use tauri::RunEvent;
use tauri::{CustomMenuItem, Manager, Menu, Submenu, Window};

const DEFAULT_PORT: u16 = 8787;
const PORT_SCAN_LIMIT: u16 = 100;
const STARTUP_ATTEMPTS: usize = 40;
const STARTUP_DELAY_MS: u64 = 250;

struct StreamlitState {
    child: Arc<Mutex<Option<Child>>>,
    active_port: Arc<Mutex<u16>>,
    last_error: Arc<Mutex<Option<String>>>,
}

fn configured_port() -> u16 {
    std::env::var("AUTOLAB_PORT")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(DEFAULT_PORT)
}

fn workspace_path() -> PathBuf {
    if let Ok(path) = std::env::var("AUTOLAB_WORKSPACE") {
        return PathBuf::from(path);
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join("HYMetaLab_autolab_workspace");
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../autolab_workspace")
}

fn script_path() -> PathBuf {
    if let Ok(path) = std::env::var("AUTOLAB_STUDIO_PATH") {
        return PathBuf::from(path);
    }

    // src-tauri -> autolab_desktop -> apps -> autolab_studio.py
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../autolab_studio.py")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from("../../autolab_studio.py"))
}

fn candidate_pythons() -> Vec<String> {
    let mut candidates = Vec::new();
    if let Ok(python) = std::env::var("AUTOLAB_PYTHON") {
        if !python.trim().is_empty() {
            candidates.push(python);
        }
    }
    candidates.push("python3".to_string());
    candidates.push("/opt/homebrew/bin/python3".to_string());
    candidates.push("/usr/local/bin/python3".to_string());
    candidates.push("/Library/Frameworks/Python.framework/Versions/Current/bin/python3".to_string());
    candidates.push("python".to_string());
    candidates
}

fn python_has_streamlit(python: &str) -> bool {
    let output = Command::new(python)
        .arg("-c")
        .arg("import streamlit")
        .output();
    match output {
        Ok(result) => result.status.success(),
        Err(_) => false,
    }
}

fn resolve_python() -> Result<String, String> {
    for candidate in candidate_pythons() {
        if python_has_streamlit(&candidate) {
            return Ok(candidate);
        }
    }
    Err(
        "Could not find a Python interpreter with Streamlit installed. \
         Install Streamlit or set AUTOLAB_PYTHON to a working interpreter."
            .to_string(),
    )
}

fn join_pythonpath(prefix: &str, existing: &str) -> String {
    #[cfg(target_os = "windows")]
    let separator = ";";
    #[cfg(not(target_os = "windows"))]
    let separator = ":";
    format!("{prefix}{separator}{existing}")
}

fn streamlit_url(port: u16) -> String {
    format!("http://127.0.0.1:{port}")
}

fn escape_js_string(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\'', "\\'")
        .replace('\n', "\\n")
        .replace('\r', "")
}

fn is_port_available(port: u16) -> bool {
    TcpListener::bind(("127.0.0.1", port)).is_ok()
}

fn choose_available_port(preferred: u16) -> Result<u16, String> {
    for offset in 0..=PORT_SCAN_LIMIT {
        let port = preferred.saturating_add(offset);
        if port > 65534 {
            break;
        }
        if is_port_available(port) {
            return Ok(port);
        }
    }

    for offset in 0..=PORT_SCAN_LIMIT {
        let port = DEFAULT_PORT.saturating_add(offset);
        if port > 65534 {
            break;
        }
        if is_port_available(port) {
            return Ok(port);
        }
    }

    Err(format!(
        "No free localhost port found near {preferred} or {DEFAULT_PORT}. \
         Close another local app or set AUTOLAB_PORT."
    ))
}

fn start_streamlit(port: u16) -> Result<Child, String> {
    let python = resolve_python()?;
    let script = script_path();
    if !script.exists() {
        return Err(format!(
            "Could not find AutoLab Studio script at {}. Set AUTOLAB_STUDIO_PATH.",
            script.display()
        ));
    }
    let run_dir = script
        .parent()
        .and_then(|parent| parent.parent())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let pythonpath = match std::env::var("PYTHONPATH") {
        Ok(existing) if !existing.trim().is_empty() => {
            join_pythonpath(&run_dir.display().to_string(), &existing)
        }
        _ => run_dir.display().to_string(),
    };

    let workspace = workspace_path();
    if let Err(err) = std::fs::create_dir_all(&workspace) {
        return Err(format!(
            "Failed to create workspace directory {}: {}",
            workspace.display(),
            err
        ));
    }

    let logs_dir = workspace.join("logs");
    if let Err(err) = std::fs::create_dir_all(&logs_dir) {
        return Err(format!(
            "Failed to create log directory {}: {}",
            logs_dir.display(),
            err
        ));
    }
    let stdout_log_path = logs_dir.join("streamlit_stdout.log");
    let stderr_log_path = logs_dir.join("streamlit_stderr.log");
    let stdout_log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&stdout_log_path)
        .map_err(|err| format!("Failed to open {}: {}", stdout_log_path.display(), err))?;
    let stderr_log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&stderr_log_path)
        .map_err(|err| format!("Failed to open {}: {}", stderr_log_path.display(), err))?;

    let mut cmd = Command::new(python);
    cmd.arg("-m")
        .arg("streamlit")
        .arg("run")
        .arg(script)
        .arg("--server.address=127.0.0.1")
        .arg(format!("--server.port={}", port))
        .arg("--server.headless=true")
        .arg("--browser.gatherUsageStats=false")
        .current_dir(&run_dir)
        .env("AUTOLAB_WORKSPACE", workspace.to_string_lossy().to_string())
        .env("PYTHONPATH", pythonpath)
        .stdout(Stdio::from(stdout_log))
        .stderr(Stdio::from(stderr_log));

    cmd.spawn()
        .map_err(|err| format!("Failed to launch Streamlit: {}", err))
}

fn wait_for_streamlit(port: u16, attempts: usize, delay_ms: u64) -> bool {
    for _ in 0..attempts {
        if TcpStream::connect(("127.0.0.1", port)).is_ok() {
            return true;
        }
        thread::sleep(Duration::from_millis(delay_ms));
    }
    false
}

fn stop_streamlit(state: &StreamlitState) {
    if let Ok(mut guard) = state.child.lock() {
        if let Some(child) = guard.as_mut() {
            if child.try_wait().ok().flatten().is_none() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
        *guard = None;
    }
}

fn current_port(state: &StreamlitState) -> u16 {
    state
        .active_port
        .lock()
        .map(|guard| *guard)
        .unwrap_or_else(|_| configured_port())
}

fn navigate_window_to_port(window: &Window, port: u16) {
    let url = streamlit_url(port);
    let _ = window.eval(&format!("window.location.replace('{}');", url));
}

fn render_startup_error(window: &Window, message: &str) {
    let escaped = escape_js_string(message);
    let js = format!(
        "document.body.style.margin='0';\
         document.body.style.fontFamily='-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif';\
         document.body.style.background='#f5f5f5';\
         document.body.style.padding='24px';\
         document.body.innerHTML='<h2 style=\"margin:0 0 12px 0;\">AutoLab backend failed to start</h2>\
         <p style=\"margin:0 0 12px 0;\">Fix the issue and use AutoLab > Restart Backend.</p>\
         <pre style=\"white-space:pre-wrap;background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;\">{}<\\/pre>';",
        escaped
    );
    let _ = window.eval(&js);
}

fn launch_backend(state: &StreamlitState) -> Result<u16, String> {
    stop_streamlit(&state);

    let preferred = configured_port();
    let port = choose_available_port(preferred)?;
    let mut child = start_streamlit(port)?;
    if !wait_for_streamlit(port, STARTUP_ATTEMPTS, STARTUP_DELAY_MS) {
        let status = child.try_wait().ok().flatten();
        if status.is_none() {
            let _ = child.kill();
            let _ = child.wait();
        }
        return Err(format!(
            "Streamlit did not become ready on {}.\nCheck logs in {}/logs/",
            streamlit_url(port),
            workspace_path().display()
        ));
    }

    if let Ok(mut guard) = state.child.lock() {
        *guard = Some(child);
    }
    if let Ok(mut guard) = state.active_port.lock() {
        *guard = port;
    }
    if let Ok(mut guard) = state.last_error.lock() {
        *guard = None;
    }
    Ok(port)
}

fn restart_streamlit(app_handle: &tauri::AppHandle) -> Result<(), String> {
    let state = app_handle.state::<StreamlitState>();
    let port = launch_backend(&state)?;

    if let Some(window) = app_handle.get_window("main") {
        navigate_window_to_port(&window, port);
    }
    Ok(())
}

fn open_path(path: &Path) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    let mut cmd = {
        let mut c = Command::new("open");
        c.arg(path);
        c
    };

    #[cfg(target_os = "windows")]
    let mut cmd = {
        let mut c = Command::new("explorer");
        c.arg(path);
        c
    };

    #[cfg(all(unix, not(target_os = "macos")))]
    let mut cmd = {
        let mut c = Command::new("xdg-open");
        c.arg(path);
        c
    };

    cmd.spawn()
        .map(|_| ())
        .map_err(|err| format!("Failed to open {}: {}", path.display(), err))
}

fn main() {
    let state = StreamlitState {
        child: Arc::new(Mutex::new(None)),
        active_port: Arc::new(Mutex::new(configured_port())),
        last_error: Arc::new(Mutex::new(None)),
    };

    if let Err(err) = launch_backend(&state) {
        eprintln!("{}", err);
        if let Ok(mut guard) = state.last_error.lock() {
            *guard = Some(err);
        }
    }

    let autolab_menu = Menu::new()
        .add_item(CustomMenuItem::new(
            "open_workspace",
            "Open Workspace Folder",
        ))
        .add_item(CustomMenuItem::new("open_runs", "Open Runs Folder"))
        .add_item(CustomMenuItem::new("reload_ui", "Reload UI"))
        .add_item(CustomMenuItem::new("restart_backend", "Restart Backend"))
        .add_item(CustomMenuItem::new("quit_app", "Quit"));

    let menu = Menu::new().add_submenu(Submenu::new("AutoLab", autolab_menu));

    let app = tauri::Builder::default()
        .manage(state)
        .menu(menu)
        .setup(|app| {
            if let Some(window) = app.get_window("main") {
                let state = app.state::<StreamlitState>();
                let startup_error = state
                    .last_error
                    .lock()
                    .ok()
                    .and_then(|guard| guard.clone());
                if let Some(error_text) = startup_error {
                    render_startup_error(&window, &error_text);
                    dialog::message(
                        Some(&window),
                        "AutoLab Startup Error",
                        error_text,
                    );
                } else {
                    navigate_window_to_port(&window, current_port(&state));
                }
            }
            Ok(())
        })
        .on_menu_event(|event| match event.menu_item_id() {
            "open_workspace" => {
                let workspace = workspace_path();
                let _ = std::fs::create_dir_all(&workspace);
                let _ = open_path(&workspace);
            }
            "open_runs" => {
                let runs_path = workspace_path().join("runs");
                let _ = std::fs::create_dir_all(&runs_path);
                let _ = open_path(&runs_path);
            }
            "reload_ui" => {
                let app_handle = event.window().app_handle();
                let state = app_handle.state::<StreamlitState>();
                navigate_window_to_port(event.window(), current_port(&state));
            }
            "restart_backend" => {
                let handle = event.window().app_handle();
                if let Err(err) = restart_streamlit(&handle) {
                    if let Some(window) = handle.get_window("main") {
                        render_startup_error(&window, &err);
                        dialog::message(
                            Some(&window),
                            "AutoLab Startup Error",
                            err.clone(),
                        );
                    }
                    eprintln!("{}", err);
                }
            }
            "quit_app" => {
                event.window().app_handle().exit(0);
            }
            _ => {}
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(|app_handle, event| {
        if let RunEvent::ExitRequested { .. } = event {
            let state = app_handle.state::<StreamlitState>();
            stop_streamlit(&state);
        }
    });
}
