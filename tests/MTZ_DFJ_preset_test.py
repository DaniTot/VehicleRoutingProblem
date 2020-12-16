
#from ..vrp import VRP
import sys
sys.path.append("..")

from vrp import VRP



k = 5  # number of vehicles

############# DFJ ############### 
vrp_dfj = VRP()

vrp_dfj.setup_preset_data(file_name="../validation_data_A/A-n32-k5.vrp",
                          number_of_vehicles=k)

vrp_dfj.gap_goal = 0.1
vrp_dfj.subtour_type = 'DFJ'
vrp_dfj.setup()
# vrp_dfj.model.Params.TimeLimit = 5*60
vrp_dfj.model.Params.Threads = 2
vrp_dfj.optimize()
vrp_dfj.visualize()


if input("continue? ").lower() not in ['y', 'yes']:
    sys.exit()

############## MTZ ############### 
vrp_mtz = VRP()

vrp_mtz.setup_preset_data(file_name="../validation_data_A/A-n32-k5.vrp",
                          number_of_vehicles=k)
vrp_mtz.gap_goal = 0.1

vrp_mtz.subtour_type = 'MTZ'
vrp_mtz.setup()
vrp_mtz.model.Params.TimeLimit = 5*60
# vrp_mtz.model.Params.Threads = 2
vrp_mtz.optimize()
vrp_mtz.visualize()


