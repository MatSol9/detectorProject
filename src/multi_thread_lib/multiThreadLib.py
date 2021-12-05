import threading
import time
from typing import List, Optional, Any
from queue import Queue


class OperationParent:
    def __init__(self, side_input: Optional[Any] = None):
        """
        A parent class for a single operation. Operation objects should implement it
        @param side_input: optional additional input used when the run() method is executed
        """
        self.side_input = side_input

    def run(self, input_object: Any) -> Any:
        """
        Run method, executed on the input_object by DataWorker, should return a type accepted by the run() method of
        the next OperationParent in OperationChain
        @param input_object: any object that will be processed by the
            OperationParent
        @return: processed input_object
        """
        return input_object

    def set_side_input(self, side_input: Any):
        """
        Updates additional input of the OperationParent
        @param side_input: any object that can serve as a side input, remember to specify types
            your OperationParent can process
        """
        self.side_input = side_input

    def get_side_input(self) -> Optional[Any]:
        """
        Returns current side input of the OperationParent
        @return: current object used as a side input
        """
        return self.side_input


"""
add_operation(operation_object: OperationParent) method is used to add consecutive data
pipeline operations. It's written so it's possible to create an OperationChain following way:
operationChain = OperationChain().add_operation(operation1).add_operation(operation2).add_operation(operation3)
"""


class OperationChain:
    def __init__(self):
        self.operations: List[OperationParent] = []

    def add_operation(self, operation_object: OperationParent):
        self.operations.append(operation_object)
        return self

    # method used by DataWorker to execute operations, don't use it
    def run_operations(self, input_object: List[Any]) -> Any:
        output_object = input_object
        for operationObject in self.operations:
            output_object = operationObject.run(output_object)
        return output_object


"""
A DataWorker object, executes the OperationChain on incoming data objects
"""


class DataWorker:
    def __init__(self, input_object: List[Queue], output_object: List[Queue], operation_chain: OperationChain):
        self.input_object = input_object
        self.output_object = output_object
        self.operation_chain = operation_chain
        self.stop_event = False

    def start(self):
        self.stop_event = False
        threading.Thread(target=self.run, args=()).start()

    def run(self):
        while not self.stop_event:
            if self.input_object:
                current_obj = []
                for input_queue in self.input_object:
                    if not input_queue.empty():
                        current_obj.append(input_queue.get())
                if current_obj:
                    for output_queue in self.output_object:
                        output_queue.put(self.operation_chain.run_operations(current_obj))

    # Use this method to stop DataWorker, waits until current OperationChain is finished
    def stop(self):
        self.stop_event = True


"""
Parent class for a data getter. Inherit it and use get_data() method for operations executed every loop iteration
"""


class GetParent:
    def __init__(self, side_input: Optional[Any] = None):
        self.side_input = side_input

    def get_data(self) -> Any:
        return None

    def stop(self):
        pass


"""
DataGetter is used to catch data input from a GetParent object
"""


class DataGetter:
    def __init__(self, output_object: List[Queue], get_parent: GetParent):
        self.output_object = output_object
        self.get_parent = get_parent
        self.stop_event = False

    def start(self):
        self.stop_event = False
        threading.Thread(target=self.run, args=()).start()

    def run(self):
        while not self.stop_event:
            current_obj = self.get_parent.get_data()
            if current_obj is not None:
                for outputQueue in self.output_object:
                    outputQueue.put(current_obj)
        self.get_parent.stop()

    def stop(self):
        self.stop_event = True


class PeriodicDataGetter:
    def __init__(self, output_object: List[Queue], get_parent: GetParent, frequency: float):
        self.output_object = output_object
        self.get_parent = get_parent
        self.stop_event = False
        self.period = 1/frequency

    def get_data(self):
        current_obj = self.get_parent.get_data()
        if current_obj is not None:
            for outputQueue in self.output_object:
                outputQueue.put(current_obj)

    def main_loop(self):
        while not self.stop_event:
            threading.Thread(target=self.get_data, args=()).start()
            time.sleep(self.period)
        self.stop_event = True

    def start(self):
        self.stop_event = False
        threading.Thread(target=self.main_loop, args=()).start()

    def stop(self):
        self.stop_event = True


"""
SinkParent object, used as an ending to a pipeline
"""


class SinkParent:
    def __init__(self, side_input: Optional[Any] = None):
        self.side_input = side_input

    def sink_data(self, input_object: list):
        pass

    def stop(self):
        pass


"""
DataSink executes the sin_data(input_object: list) function on objects incoming to each of input queues
"""


class DataSink:
    def __init__(self, input_object: List[Queue], sink_parent: SinkParent):
        self.input_object = input_object
        self.sink_parent = sink_parent
        self.stop_event = False

    def start(self):
        self.stop_event = False
        threading.Thread(target=self.run, args=()).start()

    def run(self):
        while not self.stop_event:
            if self.input_object:
                current_obj = []
                for input_queue in self.input_object:
                    if not input_queue.empty():
                        current_obj.append(input_queue.get())
                if current_obj:
                    self.sink_parent.sink_data(current_obj)
        self.sink_parent.stop()

    def stop(self):
        self.stop_event = True
