from unittest import TestCase
from queue import Queue

import src.multi_thread_lib.multiThreadLib as mtl


class Test(TestCase):
    def test_data_worker(self):
        input_queue = Queue()
        input_queue.put("1")
        input_queue.put("2")
        input_queue.put("3")
        output_queue = Queue()
        data_worker = mtl.DataWorker([input_queue], [output_queue], mtl.OperationChain()).start()
        # time.sleep(0.001)
        data_worker.stop()
        output_list = []
        while not output_queue.empty():
            output_list.append(output_queue.get())
        self.assertTrue(
            output_list.__contains__(["1"]) and output_list.__contains__(["2"]) and output_list.__contains__(["3"]))

    def test_data_getter(self):
        class TestGetObject(mtl.GetParent):
            def __init__(self):
                super().__init__(None)
                self.test_data = ["1", "2", "3"]

            def get_data(self):
                if self.test_data:
                    return self.test_data.pop()
                else:
                    return None
        output_queue = Queue()
        data_getter = mtl.DataGetter([output_queue], TestGetObject()).start()
        # time.sleep(0.001)
        data_getter.stop()
        output_list = []
        while not output_queue.empty():
            output_list.append(output_queue.get())
        self.assertTrue(output_list.__contains__("1") and output_list.__contains__("2") and output_list.__contains__("3"))

    def test_data_sink(self):
        input_queue = Queue()
        input_queue.put("1")
        input_queue.put("2")
        input_queue.put("3")
        data_sink = mtl.DataSink([input_queue], mtl.SinkParent()).start()
        # time.sleep(0.001)
        data_sink.stop()
        self.assertTrue(input_queue.empty())
