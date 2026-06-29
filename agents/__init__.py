from agents.bc import BCAgent
from agents.cdp import CDPAgent
from agents.fbrac import FBRACAgent
from agents.fql import FQLAgent
from agents.gfp import GFPAgent
from agents.ifql import IFQLAgent
from agents.iql import IQLAgent
from agents.meanflow_fql import MeanFlowFQLAgent
from agents.qam import QAMAgent
from agents.rebrac import ReBRACAgent
from agents.shortcut_fql import ShortcutFQLAgent

agents = dict(
    bc=BCAgent,
    cdp=CDPAgent,
    fbrac=FBRACAgent,
    fql=FQLAgent,
    gfp=GFPAgent,
    ifql=IFQLAgent,
    iql=IQLAgent,
    meanflow_fql=MeanFlowFQLAgent,
    qam=QAMAgent,
    rebrac=ReBRACAgent,
    shortcut_fql=ShortcutFQLAgent,
)

try:
    from agents.codac import CODACAgent
except ModuleNotFoundError as exc:
    if exc.name != 'agents.codac':
        raise
else:
    agents['codac'] = CODACAgent
