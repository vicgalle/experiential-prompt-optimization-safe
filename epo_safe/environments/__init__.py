from epo_safe.environments.side_effects import SideEffectsWrapper
from epo_safe.environments.off_switch import OffSwitchWrapper
from epo_safe.environments.absent_supervisor import AbsentSupervisorWrapper
from epo_safe.environments.boat_race import BoatRaceWrapper
from epo_safe.environments.whisky_gold import WhiskyGoldWrapper


ENV_REGISTRY = {
    "side_effects": SideEffectsWrapper,
    "off_switch": OffSwitchWrapper,
    "absent_supervisor": AbsentSupervisorWrapper,
    "boat_race": BoatRaceWrapper,
    "whisky_gold": WhiskyGoldWrapper,
}
