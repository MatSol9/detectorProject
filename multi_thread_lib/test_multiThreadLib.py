from collections import deque
from unittest import TestCase

import multi_thread_lib.multiThreadLib as mtl


class Test(TestCase):
    def test_data_worker(self):
        input_queue = deque(["1", "2", "3"])
        output_queue = deque([])
        data_worker = mtl.DataWorker([input_queue], [output_queue], mtl.OperationChain()).start()
        # time.sleep(0.001) might be needed here if tests fail
        data_worker.stop()
        self.assertTrue(output_queue.__contains__(["1"]) and output_queue.__contains__(["2"]) and output_queue.__contains__(["3"]))

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
        output_queue = deque([])
        data_getter = mtl.DataGetter([output_queue], TestGetObject()).start()
        # time.sleep(0.001) might be needed here if tests fail
        data_getter.stop()
        self.assertTrue(output_queue.__contains__("1") and output_queue.__contains__("2") and output_queue.__contains__("3"))

    def test_data_sink(self):
        input_queue = deque(["1", "2", "3"])
        data_sink = mtl.DataSink([input_queue], mtl.SinkParent()).start()
        # time.sleep(0.001) might be needed here if tests fail
        data_sink.stop()
        self.assertTrue(not input_queue)
