import lark
from src.annotation_parser.labfile_transformer import LabFileTransformer
from src.annotation_parser.rs200_dt_transformer import Rs200DtTransformer

labfile_transformer = LabFileTransformer()
labfile_parser = lark.Lark.open("./src/annotation_parser/labfile_grammar.lark", parser="lalr", start=["labfile", "chord"])

rs200_dt_transformer = Rs200DtTransformer()
rs200_dt_parser = lark.Lark.open("./src/annotation_parser/rs200_dt_grammar.lark", parser="lalr", start=["cltfile", "chord"])


def parse_annotation_file(path: str):
    with open(path) as f:
        text = f.read()
    if path.endswith(".lab") or path.endswith(".txt"):
        return labfile_transformer.transform(labfile_parser.parse(text, start="labfile"))
    elif path.endswith("_dt.clt"):
        return rs200_dt_transformer.transform(rs200_dt_parser.parse(text, start="cltfile"))


def parse_lab_annotation(annotation: str):
    return labfile_transformer.transform(labfile_parser.parse(annotation, start="chord"))
