class DocumentElement(object):
    """
        Class that defines a generic template for each element of Document object
    """
    __slots__ = ("__document", "__idx")

    def __init__(self, document):
        self.__document = document
        self.__idx = -1

    """
        Getters and setters
    """

    @property
    def document(self):
        return self.__document

    @document.setter
    def document(self, value):
        self.__document = value

    @property
    def idx(self):
        return self.__idx

    @idx.setter
    def idx(self, value):
        self.__idx = value


class Position(object):
    """
        The class that represents chars offsets for document elements that have a reference to text content
    """
    __slots__ = ("__start", "__end")

    def __init__(self, start, end):
        self.start = start
        self.end = end

    @property
    def start(self):
        return self.__start

    @start.setter
    def start(self, value):
        self.__start = value

    @property
    def end(self):
        return self.__end

    @end.setter
    def end(self, value):
        self.__end = value


class Extraction(DocumentElement):
    """
        The class that represents the Cogito Extraction Object
    """

    __slots__ = ("__namespace", "__template", "__fields")

    def __init__(self, document, namespace, template, fields=None):
        super().__init__(document)
        #
        self.__namespace = namespace
        self.__template = template
        self.__fields = fields if fields is not None else []

    @property
    def namespace(self):
        return self.__namespace

    @namespace.setter
    def namespace(self, value):
        self.__namespace = value

    @property
    def template(self):
        return self.__template

    @template.setter
    def template(self, value):
        self.__template = value

    @property
    def fields(self):
        return self.__fields

    @fields.setter
    def fields(self, value):
        self.__fields = value

    def add_field(self, name, val, positions_dicts, rules_indexes=None):
        ef = Extraction.ExtractionField(name=name, val=val, rules_indexes=rules_indexes)
        for pd in positions_dicts:
            ef.add_position(start=pd["start"], end=pd["end"])
        #
        self.__fields.append(ef)

    class ExtractionField(object):

        __slots__ = ("__name", "__val", "__positions", "__rules_indexes")

        def __init__(self, name, val, positions=None, rules_indexes=None):
            self.__name = name
            self.__val = val
            self.__positions = positions if positions is not None else []
            self.__rules_indexes = rules_indexes

        @property
        def name(self):
            return self.__name

        @name.setter
        def name(self, value):
            self.__name = value

        @property
        def val(self):
            return self.__val

        @val.setter
        def val(self, value):
            self.__val = value

        @property
        def positions(self):
            return self.__positions

        @positions.setter
        def positions(self, value):
            self.__positions = value

        def add_position(self, start, end):
            pos = Position(start=start, end=end)
            self.__positions.append(pos)

        @property
        def rules_indexes(self):
            return self.__rules_indexes

        @rules_indexes.setter
        def rules_indexes(self, value):
            self.__rules_indexes = value
