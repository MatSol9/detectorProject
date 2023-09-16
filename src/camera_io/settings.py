from data_model.dataModel import Config


class Settings:
    def __init__(self):
        self.config = Config()
        temp_objects = self.config.get_objects()
        self.indexes: List[int] = list(temp_objects.keys())
        self.options = None
        self.tags_index = {}
        self.tags = []
        for index in self.indexes:
            self.tags_index[temp_objects.get(index).get("tag_id")] = index
            self.tags.append(temp_objects.get(index).get("tag_id"))
