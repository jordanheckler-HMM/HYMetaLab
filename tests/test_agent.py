from sim.agent import Agent


def test_agent_lifecycle():
    a = Agent(agent_id=1, energy=10.0)
    assert a.is_alive()
    a.step(energy_cost=1.0)
    assert a.age == 1
    assert a.energy == 9.0
    a.move(1, 0, cost=2.0)
    assert a.x == 1 and a.y == 0
    assert a.energy == 7.0
    # deplete energy
    a.step(energy_cost=10.0)
    assert a.energy == 0.0
    assert not a.is_alive()
