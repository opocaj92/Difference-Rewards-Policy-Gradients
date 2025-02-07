from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .a2c_learner import A2CLearner
from .pg_learner import PGLearner
from .drreinforcer_learner import DrReinforceRLearner
from .drppo_learner import DrPPOLearner
from .dra2c_learner import DRA2CLearner
from .colby_learner import ColbyLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["a2c_learner"] = A2CLearner
REGISTRY["pg_learner"] = PGLearner
REGISTRY["drreinforcer_learner"] = DrReinforceRLearner
REGISTRY["drppo_learner"] = DrPPOLearner
REGISTRY["dra2c_learner"] = DRA2CLearner
REGISTRY["colby_learner"] = ColbyLearner
