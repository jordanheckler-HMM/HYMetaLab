from pydantic import BaseModel, Field, conlist


class Sweep(BaseModel):
    epsilon: conlist(float, min_items=1)
    seeds: conlist(int, min_items=1)
    shock: conlist(float, min_items=1)
    agents: list[int] | None = None


class StudyConfig(BaseModel):
    id: str
    hypothesis: str
    sweep: Sweep
    metrics: conlist(str, min_items=1)
    exports: conlist(str, min_items=1)
    adapter: str = Field(default="adapters/sim_adapter_safe.py")
