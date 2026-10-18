# Must bind `node_scout` in Nuke's __main__: the panel's PyCustom_Knob evaluates
# "node_scout.ui.nuke.prediction_panel...()" there, and a from-import leaves it unbound.
import node_scout.nuke_startup

node_scout.nuke_startup.on_startup()
