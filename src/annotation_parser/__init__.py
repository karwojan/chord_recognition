import lark
from src.annotation_parser.labfile_transformer import LabFileTransformer

labfile_transformer = LabFileTransformer()
labfile_parser = lark.Lark.open("./src/annotation_parser/labfile_grammar.lark", parser="lalr", start="labfile")


def parse_labfile(path):
    with open(path) as f:
        text = f.read()
    return labfile_transformer.transform(labfile_parser.parse(text))
