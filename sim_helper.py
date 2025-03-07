
# Sim Helper 
# Dependencies: snn_helper, memchip_python
# Helper functions and classes for running sims

import sys
sys.path.append('../analysis/sofie_python')


import itertools
import time

from memchip_python import *
from snn_helper import *
from boltzmann_helper import * 


##################################################################################
############## Memchip Functions #################################################
##################################################################################



def get_default_memchip(size,path):

    if size == 200:
        ###########200 board ###########################
        input_groups = [15, 1320, 4 ,1319]
        output_groups = [34]
        seed=382
        ER=2000000
        IR=33330
        size=200
    elif size == 400:
        ############ 400 board, 8 contacts #####################
        # input_groups = [5, 2378,5987,6076,6140,2766,222,35]
        input_groups = [5, 2378,5987,6076,6140,2766,222,6]
        output_groups = [6575]
        seed = 223
        ER = 500000
        IR = 5000
        size = 400
    memchip = init_memchip(seed, ER, IR, size, input_groups, output_groups)

    memchip.input_T = 1000
    memchip.voltage = 3 # high voltage
    memchip.timestep = 0

    ################### other params #######################################
    memchip.output_dir = Path(os.getcwd()).parent / path
    asg_files_dir = memchip.output_dir
    memchip.total_timesteps = 1000000000
    n = 0
    memchip._prev_length_mod = np.zeros((memchip._ntunnels,))  # zero initially
    memchip.kappas = np.array([])
    memchip.mus = np.array([])
    memchip.multicontact = 1
    return memchip


####### get sim ###############
def get_sim(voltage,input_groups,path,low=0,input_T=10000,size=400,inputs_num=4,control_voltages=[0,0,1,1],seed=0,output_groups=[],output_v_groups=[],**kwargs):
    """ make memchip and sim object 
    input groups = any groups you want to apply a voltage to -> include control electrodes and input electrodes
    output_groups=group to be set at 0V -> by default, will find an unconnected one
    output_v_groups=groups to record floating voltage from
    """
    ## get memchip with corresponding seed so we can find output groups that are unconnected
    memchip = init_memchip(seed, 1, 1, size, input_groups, [5],input_groups_manual = input_groups[0:inputs_num])
    unconnected = np.nonzero(memchip._connected_bool == False)

    if size == 400:
        output_groups = [unconnected[0][0]] if len(output_groups)==0 else output_groups
        seed = 233 if seed == 0 else seed
        ER = 500000
        IR = 5000
    elif size == 200:
        # default = 382
        output_groups = [unconnected[0][0]] if len(output_groups)==0 else output_groups
        seed = 382 if seed == 0 else seed
        ER=2000000
        IR=33330
    memchip = init_memchip(seed, ER, IR, size, input_groups, output_groups,input_groups_manual = input_groups[0:inputs_num],**kwargs)

    memchip.input_T = input_T
    memchip.voltage = voltage # high voltage
    memchip.timestep = 0
    memchip.output_v_groups=output_v_groups

    memchip.output_dir = Path(os.getcwd()).parent / path

    d = sim(memchip,low,control_voltages=control_voltages)
    return d



#############################################################################
##################### Binary patterns #######################################
#############################################################################

def get_possible_patterns(bits,high_v,low_v=0):
    """ 
    return all possible patterns for a number of bits. Pass in high V encoding 1 and optionally low V encoding 0.
    """
    return np.array(list(itertools.product([low_v, high_v], repeat=bits)))

def reset(timesteps, memchip,high):
    '''reset board by setting all inputs high and outputs low'''
    for step in range(timesteps):
        memchip.write([[high, high, high, high, 0 ,0, 0, 0]], determine_tool(self, 'da'))
        memchip.timestep += 1
    return memchip



#############################################################################
################### Sim Class ###############################################
#############################################################################

class sim:
    """ 
    create instance of this class to run sim
    """
    def __init__(self,memchip,low=0,control_voltages = [0,0,1,1]):
        memchip.total_timesteps = 9000000000
        memchip._prev_length_mod = np.zeros((memchip._ntunnels,))  # zero initially
        memchip.kappas = np.array([])
        memchip.mus = np.array([])
        memchip.multicontact = 1
        self.low = low
        self.memchip = memchip
        self.size = memchip.width
        self.control_voltages = np.array(control_voltages)

    def apply_input(self,input,input_T=0):
        """
        INPUT should be shape (#electrodes)
        if element in input is 0, write 0 to electrode unless the LIF neuron attached to electrode is spiking, then apply Vhigh
        if element in input is not 1, write Vhigh * element to electrode
        """
        self.memchip.calculate_matrix=True
        if input_T == 0:
            input_T = self.memchip.input_T
        for i in range(input_T):    
            # write to memchip
            self.memchip.write([input], determine_tool(self.memchip, 'da'))
            self.memchip.timestep += 1


    def apply_sequences(self,num_bits,sema,low=0,num_instances=12,init=False,overthreshold_scale_factor=1,info=False,maps=False,shuffle=True):
        """
        apply all possible patters with equal frequency to board. 
        assumes board has 8 electrodes and the first 4 are inputs
        num_instances - number of times to show board all possible patterns
        """

        #################### sequences ###############################
        sequences=get_possible_patterns(num_bits,self.memchip.voltage,self.low)
        # inputs are first 4 going clockwise
        sequences=np.c_[sequences,np.resize(self.control_voltages,(2**num_bits, len(self.control_voltages)))]

        # distribute inputs around board
        # sequences=np.c_[np.ones(len(sequences))*high,sequences[:,0],np.zeros(len(sequences)),sequences[:,1],np.zeros(len(sequences)),sequences[:,2],np.ones(len(sequences))*high,sequences[:,3]]
        if info:
            print("sequences are:")
            print(sequences)
            self.draw_electrodes()
            self.draw_all()
            self.print_info()

        high=self.memchip.voltage

        #################### initialise board ##################################
        if init:
            # print('starting')
            start = time.perf_counter()
            num = len(sim.memchip.input_groups)
            input_init = [ overthreshold_scale_factor*high*(i<num_bits) for i in range(num)]
            sim.apply_input(input_init,40)
            finish = time.perf_counter()
            # print(f"Initialised 200000 steps in {(finish - start) / 60 :0.4f} mins")
        else:
            print("not initialised")

        ################### patterns #########################################
        # raw_events = []
        for i in range(num_instances):
            for p in range(2**num_bits):
                # print(f'input is {sequences[p,:]}')
                self.apply_input(sequences[p,:])
                # weird fix - can't just iterate over seq...
                # sim.draw_current(str(seq))
                if maps:
                    self.draw_currents('_'+str(sequences[p,:]) + str(i))
                    self.draw_potentials('_'+str(sequences[p,:]) + str(i))

            if info:
                print('done _ ' + str(i)) 
            if shuffle:
                np.random.shuffle(sequences)
        if sema is not None:
            sema.release()

    def draw_electrodes(self):
        """ if inputs manual defined, will draw """
        # draw_board(self.memchip,"centroids",self.memchip.output_dir/'electrode_groups')
        draw_board(self.memchip,"current_map",self.memchip.output_dir/ str(self.memchip.filename_info + 'electrode_groups_2'))

    def draw_all(self):
        """ if inputs manual defined, will draw """
        draw_board(self.memchip,"centroids",self.memchip.output_dir/str(self.memchip.filename_info + 'electrode_groups'))
        draw_board(self.memchip,"centroids_clean",self.memchip.output_dir/self.memchip.output_dir/str(self.memchip.filename_info + 'board_centroids'))
        draw_board(self.memchip,"particles",self.memchip.output_dir/self.memchip.output_dir/str(self.memchip.filename_info + 'particles'))
        draw_board(self.memchip,"current_map",self.memchip.output_dir/'electrode_groups_2')

    def draw_potentials(self,name=''):
        draw_board(self.memchip,"particles_clean",self.memchip.output_dir/('potentials_'+name))

    def draw_currents(self,name=''):
        draw_board(self.memchip,"current_map",self.memchip.output_dir/('current_'+name))

    def switch_density(self,title='switch_density',save=False):
        """ todo """
        b = self.memchip
        fig, ax = plt.subplots()
        ax.set_xlim((-5, b.width + 5))
        ax.set_ylim((-5, b.height + 5))

        events_at_each_site = {(35, 93): 2, (93, 813): 2, (493, 929): 2, (493, 1310): 2, (608, 616): 2, (1507, 1903): 2, (2348, 2632): 2, (2766, 3132): 2, (2802, 3729): 2, (2893, 2922): 2, (3110, 3760): 2, (5354, 5518): 2, (5518, 6319): 2}
        b.draw(ax, events_at_each_site, style='switch_density')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()
        print("Drawn")
        if save:
            plt.savefig(str(b.output_dir) + '/switch_density_plot.png',
                        bbox_inches='tight', dpi=600)
        else:
            plt.show()

        # draw_board(self.memchip,"switch_density",self.memchip.output_dir/title,sw_events)

    def get_isolated_groups(self):
        return self.memchip.isolated_groups

    def print_info(self):
        print(f'output dir = {self.memchip.output_dir}')
        print('groups: ')
        print(self.memchip.input_groups, self.memchip.output_groups)
        print(f'seed = {self.memchip.rand_seed}')

#############################################################################
################### SNN sim #################################################
#############################################################################

class snn_sim:
    """ 
    LIF neuron attached to each electrode 
    When neuron fires, connceted electrode is V_high for spike_length timesteps
    This can cause other electrodes to be high
    """
    def __init__(self,memchip,neurons,spike_count=np.array([0,0,0,0]),spike_length=100,Vhigh=1,Vlow=0):
        self.memchip = memchip
        if len(spike_count < len(memchip.input_groups)):
            spike_count = np.zeros(len(memchip.input_groups))
        self.spike_count=spike_count
        self.spike_length=spike_length
        self.Vhigh=Vhigh
        self.Vlow=Vlow
        self.neurons = neurons
        self.spiked = []

        # snn file
        filename = memchip.output_dir / 'R{}_V_{}_X{}_Y{}_{}_membrane_potential_and_spiking_events.csv'.format(memchip.rand_seed,memchip.voltage, memchip.width, memchip.height, memchip.total_timesteps)
        snn_file = open(filename, 'w', newline='\n')
        for i in range(memchip.input_count):
            snn_file.write("Vmembrane_N_" + str(memchip.input_groups[i]) + ",")
            snn_file.write("Spikes_N_" + str(memchip.input_groups[i]) + ",")
            snn_file.write("Input_" + str(memchip.input_groups[i]) + ",")    # external input
        snn_file.write('\n')
        self.snn_file=snn_file

# might need __ b/c have to overwrite from sim class 
    def apply_input(self,timesteps,input):
        """
        INPUT should be shape (#electrodes)
        applies input to memchip while in the SNN configuration
        if element in input is 0, write 0 to electrode unless the LIF neuron attached to electrode is spiking, then apply Vhigh
        if element in input is not 1, write Vhigh * element to electrode
        """
        prev_input_currents = 0
        for i in range(timesteps):    
            # write to memchip
            memchip_input = self.spike_count > 0
            memchip_input=memchip_input.astype(int)
            memchip_input[input==1] = int(self.Vhigh)    #int(i) * self.Vhigh
            memchip_input=memchip_input.astype(int)

            input_to_neurons = snn_sim.get_input(self.memchip.input_currents,prev_input_currents,self.spiked)
            self.update_neurons(input_to_neurons,input)

            self.memchip.calculate_matrix=True
            self.memchip.write([memchip_input], determine_tool(self.memchip, 'da'))
            self.memchip.timestep += 1
            prev_input_currents=self.memchip.input_currents
            
            self.spike_count[self.spiked] = self.spike_length
            self.spike_count = self.spike_count - 1

    def update_neurons(self, currents,input):
        """ 
        updates each neuron in neuron_list by applying current for 1 timestep
        writes results to file 
        returns list of neurons that spiked
        """
        self.spiked = []
        for neuron,current, num in zip(self.neurons,currents,range(len(self.neurons))):
            self.snn_file.write('{:.16e}, '.format(neuron.membrane_potential))
            self.snn_file.write('{}, '.format(neuron.spiked))
            self.snn_file.write('{}, '.format(input[num]))
            if neuron.spiked:
                self.spiked.append(num)
            neuron.update_membrane_potential(current)
        self.snn_file.write('\n')

    def close_snn_file(self):
        """ call when sim is finished"""
        self.snn_file.close()


    @classmethod   
    def get_input(cls,current,prev_current,spiked):
    # print(abs(current - prev_current))
    # print(np.where(abs(current - prev_current)>threshold,1,0))
    # -1*current abs(current - prev_current)>threshold
        out = np.where(-1*current>0,-1*current,0)
        out[spiked] = 0
        return (out)



####################################################################
################## Int Factorization ###############################
################## PBIT SIM ########################################
####################################################################

class pbit_sim:
    """ 
    Each PNN is a PBit 
    This class is for factorisation of integer F with number of bits in x, y = p+1,q+1
    dependencies: boltzmann_helper
    """
    def __init__(self,path,p=2,q=2,F=35,seeds=[382],alpha=10,offsets = [15.5], threshold = 10, tau = 100,log_all=False,coupled=True,type_='async',conductance_based=False):
        """" 
        make a PNN pbit for each bit in X and Y
        path is a relative path in .../results to save the data 
        seeds is an array of seeds to construct the pbits. If seeds = [], all the networks have the same default seed, 382
        if pass in seeds, also pass in array of offsets to shift the input voltage range. The required value will depend on the seed.
        type can be async, async_tref or sequential. The update functions are then evolve continuously, evolve_continuously_tref and evolve single pbit.
        conductance_based - by default not conductance based (current based), threshold is delta CURRENT. If true, threshold is delta conductance. To apply input funciton need to pass in previous v_in
        """


        if log_all: # log voltages for every single pbit
            self.determine = 'da'
        else: # only log state
            self.determine='n'

        # coupling depends on number of bits and int to factorise F
        num_pbits = p+q
        self.p = p
        self.q = q
        int_factorisation = IntF(p,q,F)
        self.coupling = int_factorisation.get_coupling()

        # randomly initialise state
        self.state=np.array([np.random.randint(100) > 50 for i in range(num_pbits)])
        self.state_count = np.array([0 for i in range(num_pbits)])

        # params 
        self.alpha=alpha
        self.threshold = threshold
        self.tau = tau
        self.coupled = coupled

        # offsets and seeds 
        assert len(offsets) == len(seeds), 'Need an offset for each seed'
        assert num_pbits == len(offsets ) or len(offsets) == 1, 'p and q do not match number of seeds and offsets'
        if len(seeds) == 1: # all the seeds and offsets are the same
            self.offsets = [offsets[0] for i in range(num_pbits)]
            self.seeds = [seeds[0] for i in range(num_pbits)]
        else: # specified offsets for each seed
            self.offsets = offsets
            self.seeds = seeds


        # initialize and make boards, add to self.boards object. 
        self.boards = []
        pbits_str = ''
        for i, seed in enumerate(self.seeds):
            if i < p:
                name = 'x'+str(i+1)
            else:
                name = 'y' + str(i-p+1)
            pbits_str += name
            pbits_str += ','
            board = init_memchip(seed, 2000000, 33330, 200, filename_info=name)
            board.input_T=0
            board.voltage = 3 
            board.timestep = 0
            board.total_timesteps = 10000000000000000
            board.output_dir = Path(os.getcwd()).parent / Path('results') / Path('pbit') / Path(path)
            board._prev_length_mod = np.zeros((board._ntunnels,))  # zero initially
            board.kappas=np.array([])
            board.mus=np.array([])
            #  initialize boards 
            init = np.random.randint(10) + 10
            for step in range(10000):
                board.write([[init]], determine_tool(board, self.determine))
                board.timestep +=1
            self.boards.append(board)
        self.type=type_
        self.conductance_based=conductance_based

        # state file
        filename = board.output_dir / f'pbit_state_scale_factor_{self.alpha}_seed_{self.seeds}_offset_{self.offsets}_threshold_{self.threshold}_coupled_{self.coupled}_type_{self.type}.csv'
        pbit_file = open(filename, 'w', newline='\n')
        pbit_file.write("iteration,")
        pbit_file.write(pbits_str)
        pbit_file.write('\n')
        self.pbit_file=pbit_file

    def evolve_single_pbit(self):
        """
        single pbit case
        """
        pass

    def evolve_continuously(self,timesteps):
        """
        async mode
        instead of applying V_in to each PNN sequentially, apply V_in to every PNN each timestep. 
        If PNN p1 spiked in the last 100 timesteps (sample interval) then the state of x1 is 1, otherwise 0
        """

        for t in range(timesteps): 
            spike_vec = []           
  
            X_ = [1 if i == 0 else self.state[i-1] for i in range(self.p+1)]
            Y_ = [1 if i == 0 else self.state[i+self.p-1] for i in range(self.q+1)]
            for i, pbit in enumerate(self.boards):
                f = self.coupling[i]     
                spike_vec.append(pbit_sim.apply_input(pbit,f(X_,Y_),self.offsets[i],self,steps=1,determine_tool_str=self.determine))

            # evolve state - if spiked, set pbit state to 1 and reset count. If count > 100, set state to 0
            self.state_count = self.state_count + 1

            # update state
            for i, el in enumerate(self.state_count):
                if el > self.tau:
                    self.state[i] = 0
                if spike_vec[i] == 1:
                    self.state[i]=1
                    self.state_count[i]=0

            str_ = ''
            str_ += str(self.boards[0].timestep)
            for el in self.state:
                str_+= ','
                str_ += str(int(el))
            str_+='\n'
            self.pbit_file.write(str_)
            
            # write state to output file every tau timesteps
            if t % self.tau == 0:
                print(t)


    def evolve_continuously_tref(self,timesteps):
        """
        alternative to async mode
        apply V_in to every PNN each timestep. 
        When PNN spikes, set state to 1
        For Tau timesteps, the state stays at 1. But no input applied to PNN because it is refractory period
        After this time, the state goes back to 0
        If PNN p1 spiked in the last 100 timesteps (sample interval) then the state of x1 is 1, otherwise 0
        """
        for t in range(timesteps): 
            spike_vec = []           
  
            X_ = [1 if i == 0 else self.state[i-1] for i in range(self.p+1)]
            Y_ = [1 if i == 0 else self.state[i+self.p-1] for i in range(self.q+1)]
            for i, pbit in enumerate(self.boards):
                if self.state_count[i] > self.tau: # only apply input if longer than t_ref since last spike
                    f = self.coupling[i]     
                    spike_vec.append(pbit_sim.apply_input(pbit,f(X_,Y_),self.offsets[i],self,steps=1,determine_tool_str=self.determine))
                else: # tref has not passed, can't spike
                    spike_vec.append(0)

            # evolve state - if spiked, set pbit state to 1 and reset count. If count > 100, set state to 0

            # add case to exclude spikes detected because Vin has just changed
            input_just_changed=False
            for count in self.state_count:
                if count == self.tau + 1: # one neuron just dropped to 0
                    input_just_changed=True
                elif count == 0: # one neuron just spiked
                    input_just_changed=True
            
            self.state_count = self.state_count + 1

            # update state
            for i, el in enumerate(self.state_count):
                if el > self.tau:
                    self.state[i] = 0
                if not input_just_changed:
                    if spike_vec[i] == 1:
                        self.state[i]=1
                        self.state_count[i]=0

            str_ = ''
            str_ += str(self.boards[0].timestep)
            for el in self.state:
                str_+= ','
                str_ += str(int(el))
            str_+='\n'
            self.pbit_file.write(str_)
            
            # write state to output file every tau timesteps
            if t % self.tau == 0:
                print(t)

    @classmethod   
    def apply_input(cls,board,input,offset,sim,steps=100,determine_tool_str = 'n',prev_v = -1):
        """"
        scales and shifts input I_i
        if you want to write out inputs etc, set deterimine_tool_str to 'da' for datfiles
        returns true if the network spiked
        """
        spiked=0
        if sim.coupled:
            input = input*sim.alpha + offset # scale factor
            input = max(0, input)
        else: 
            input = offset
        board.calculate_matrix=1
        prev_i = sum(board.output_currents)
        for step in range(steps):
            board.write([[input]], determine_tool(board, determine_tool_str))
            board.timestep +=1
            delta_i = abs(prev_i - sum(board.output_currents))
            if delta_i > sim.threshold:
                spiked = 1 
        return spiked