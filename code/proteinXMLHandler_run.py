import xml.sax
import os
from ProteinXMLHandler import ProteinXMLHandler
import gzip

data_path = "/home/damian/Documents/L3S/projects"
file_in ="match_complete.xml.gz"
# create an XMLReader
parser = xml.sax.make_parser()
# turn off namepsaces
parser.setFeature(xml.sax.handler.feature_namespaces, 0)

# override the default ContextHandler
Handler = ProteinXMLHandler()
parser.setContentHandler(Handler)
parser.parse(gzip.open(os.path.join(data_path,file_in), 'rt'))