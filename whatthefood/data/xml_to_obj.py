import os
import xml.etree.ElementTree as XMLElementTree


def parse_dir(dir_name, recursive=True):
    elements = [
        parse_file(os.path.join(dir_name, filename))
        for filename in os.listdir(dir_name) if os.path.splitext(filename)[1] == '.xml'
    ]
    if recursive:
        for subdir in os.listdir(dir_name):
            if os.path.isdir(os.path.join(dir_name, subdir)):
                elements.extend(parse_dir(os.path.join(dir_name, subdir), recursive=True))

    return elements


def parse_file(xml_filename):
    file = XMLElementTree.parse(xml_filename)
    obj = file.getroot()

    assert obj.tag == 'annotation'

    return Annotation(obj)


class Object(object):
    def __init__(self, xml_obj=None):
        self.center = None
        self.size = None
        self.label = None

        if xml_obj:
            self.parse(xml_obj)

    def parse(self, xml_obj):
        bb = xml_obj.find('bndbox')
        xmin = int(bb.find('xmin').text)
        xmax = int(bb.find('xmax').text)
        ymin = int(bb.find('ymin').text)
        ymax = int(bb.find('ymax').text)

        self.center = (
            (xmin + xmax) // 2,
            (ymin + ymax) // 2,
        )
        self.size = (
            xmax - xmin,
            ymax - ymin,
        )

        self.label = xml_obj.find('name').text


class Annotation(object):
    def __init__(self, xml_obj=None):
        self.img_name = None
        self.img_size = None
        self.objects = None

        if xml_obj:
            self.parse(xml_obj)

    def parse(self, xml_obj):
        self.img_name = xml_obj.find('filename').text
        size = xml_obj.find('size')
        self.img_size = (
            int(size.find('width').text),
            int(size.find('height').text),
            int(size.find('depth').text),
        )

        self.objects = [Object(o) for o in xml_obj.findall('object')]
