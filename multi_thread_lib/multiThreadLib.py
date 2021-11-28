import threading

from typing import List
from collections import deque

"""
A parent class for a single operation. Operation object should implement it
"""


class OperationParent:
    def __init__(self, side_input=None):
        self.side_input = side_input

    # Run method, executed on the input_object by DataWorker, should return a type accepted by the run() method of the
    # next OperationParent in OperationChain
    def run(self, input_object):
        return input_object


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
    def run_operations(self, input_object: list):
        output_object = input_object
        for operationObject in self.operations:
            output_object = operationObject.run(output_object)
        return output_object


"""
A DataWorker object, executes the OperationChain on incoming data objects
"""


class DataWorker:
    def __init__(self, input_object: List[deque], output_object: List[deque], operation_chain: OperationChain):
        self.input_object = input_object
        self.output_object = output_object
        self.operation_chain = operation_chain
        self.stop_event = threading.Event()

    def start(self):
        threading.Thread(target=self.run, args=()).start()
        return self

    def run(self):
        while not self.stop_event.is_set():
            if self.input_object:
                current_obj = []
                for input_queue in self.input_object:
                    if input_queue:
                        current_obj.append(input_queue.popleft())
                if current_obj:
                    for output_queue in self.output_object:
                        output_queue.append(self.operation_chain.run_operations(current_obj))

    # Use this method to stop DataWorker, waits until current OperationChain is finished
    def stop(self):
        self.stop_event.set()


"""
Parent class for a data getter. Inherit it and use get_data() method for operations executed every loop iteration
"""


class GetParent:
    def __init__(self, side_input=None):
        self.side_input = side_input

    def get_data(self):
        return None

    def stop(self):
        pass


"""
DataGetter is used to catch data input from a GetParent object
"""


class DataGetter:
    def __init__(self, output_object: List[deque], get_parent: GetParent):
        self.output_object = output_object
        self.get_parent = get_parent
        self.stop_event = threading.Event()

    def start(self):
        threading.Thread(target=self.run, args=()).start()
        return self

    def run(self):
        while not self.stop_event.is_set():
            current_obj = self.get_parent.get_data()
            if current_obj is not None:
                for outputQueue in self.output_object:
                    outputQueue.append(current_obj)
        self.get_parent.stop()

    def stop(self):
        self.stop_event.set()


"""
SinkParent object, used as an ending to a pipeline
"""


class SinkParent:
    def __init__(self, side_input=None):
        self.side_input = side_input

    def sink_data(self, input_object: list):
        pass

    def stop(self):
        pass


"""
DataSink executes the sin_data(input_object: list) function on objects incoming to each of input queues
"""


class DataSink:
    def __init__(self, input_object: List[deque], sink_parent: SinkParent):
        self.input_object = input_object
        self.sink_parent = sink_parent
        self.stop_event = threading.Event()

    def start(self):
        threading.Thread(target=self.run, args=()).start()
        return self

    def run(self):
        while not self.stop_event.is_set():
            if self.input_object:
                current_obj = []
                for input_queue in self.input_object:
                    if input_queue:
                        current_obj.append(input_queue.popleft())
                if current_obj:
                    self.sink_parent.sink_data(current_obj)
        self.sink_parent.stop()

    def stop(self):
        self.stop_event.set()
