from tminterface.interface import TMInterface
import copy

class Worker():
    def __init__(self):
        self.have_current_run = False

        self.should_client_work = False #variabila care indica DACA AR trebui sa ruleze clientul (ie e ceva in stiva)
        self.is_client_redoing = True #variabila care indica ralanti pentru simulari
        self.input_stack = [] #stiva cu [(input ce se doreste a fi rulat, scorul inputului dupa rulare)].
        self.input_stack_index = -1 #indica care input trebuie rulat acum.

        self.debug = 0

        pass

    def add_input_array_to_stack(self, input):
        self.input_stack.append([copy.deepcopy(input), None])
        self.input_stack_index += 1
        if not self.should_client_work:
            self.should_client_work = True

    #se apeleaza la sfarsitul unei simulari.
    #(la sfarsitul on_checkpoint_count_changed, trb actualizat self.current_fitness_score)
    #!! aici trb sa apelezi write_input_array_to_EventBufferData (daca mai ai ceva in stiva)
    def process_input_stack(self, iface: TMInterface):
        self.input_stack[self.input_stack_index][1] = 0
        self.input_stack_index -= 1
        if self.input_stack_index < 0:
            self.should_client_work = False
            self.is_client_redoing = True

    def clear_stack(self):
        self.input_stack = []
        self.input_stack_index = -1

    def main_loop(self):
        if not self.should_client_work:
            if len(self.input_stack) == 0:
                #trebuie dat de munca clientului.
                #trebuie bagate toate inputurile noi in acelasi timp in self.input_stack.
                if self.debug == 0:
                    self.add_input_array_to_stack(([0] * 434, [1] * 434, [0] * 434))
                    self.debug = 1
            else:
                #daca sunt aici inseamna ca clientul a terminat munca pe care i-am dat-o, trebuie analizata,
                #trebuie calculate procentajele noi si trebuie golita stiva
                self.clear_stack()    
        else:
            #lasa clientul sa lucreze ce i-ai dat.
            pass

        pass