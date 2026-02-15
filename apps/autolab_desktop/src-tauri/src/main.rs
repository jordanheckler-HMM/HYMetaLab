#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use std::net::TcpStream;
use std::path::Path;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use tauri::{CustomMenuItem, Manager, Menu, Submenu};
use tauri::RunEvent;

const DEFAULT_PORT: u16 = 8787;

struct StreamlitState {
    child: Arc<Mutex<Option<Child>>>,
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

fn start_streamlit() -> Result<Child, String> {
    let python = std::env::var("AUTOLAB_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let script = script_path();
    if !script.exists() {
        return Err(format!(
            "Could not find AutoLab Studio script at {}. Set AUTOLAB_STUDIO_PATH.",
            script.display()
        ));
    }

    let port = configured_port();
    let workspace = workspace_path();
    if let Err(err) = std::fs::create_dir_all(&workspace) {
        return Err(format!(
            "Failed to create workspace directory {}: {}",
            workspace.display(),
            err
        ));
    }

    let mut cmd = Command::new(python);
    cmd.arg("-m")
        .arg("streamlit")
        .arg("run")
        .arg(script)
        .arg("--server.address=127.0.0.1")
        .arg(format!("--server.port={}", port))
        .arg("--server.headless=true")
        .arg("--browser.gatherUsageStats=false")
        .env("AUTOLAB_WORKSPACE", workspace.to_string_lossy().to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

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

fn restart_streamlit(app_handle: &tauri::AppHandle) -> Result<(), String> {
    let state = app_handle.state::<StreamlitState>();
    stop_streamlit(&state);
    let child = start_streamlit()?;
    if let Ok(mut guard) = state.child.lock() {
        *guard = Some(child);
    }

    let port = configured_port();
    let _ = wait_for_streamlit(port, 40, 250);

    if let Some(window) = app_handle.get_window("main") {
        let _ = window.eval("window.location.reload();");
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
    let port = configured_port();

    let child = match start_streamlit() {
        Ok(process) => Some(process),
        Err(err) => {
            eprintln!("{}", err);
            None
        }
    };

    let state = StreamlitState {
        child: Arc::new(Mutex::new(child)),
    };

    // Give Streamlit a short startup window before the webview loads.
    let _ = wait_for_streamlit(port, 40, 250);

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
                let _ = event.window().eval("window.location.reload();");
            }
            "restart_backend" => {
                let handle = event.window().app_handle();
                if let Err(err) = restart_streamlit(&handle) {
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
