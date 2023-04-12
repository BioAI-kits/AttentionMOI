import sys
import argparse

class DefaultHelpArgParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write("Error: %s\n" %message)
        self.print_help()
        sys.exit(1)

def process_args(sysargs=None):
    cmds = {
        "version": do_version,
        "predict": do_predict,
        "train": do_train,
    }

    parser = DefaultHelpArgParser(description="Run Moanna")

    parser.add_argument("-v", "--version", action="store_true")

    subparsers = parser.add_subparsers(dest="command")

    add_predict_args(subparsers.add_parser("predict", help="Run Moanna Pre-trained prediction"))
    add_train_args(subparsers.add_parser("train", help="Run Moanna training"))

    args = parser.parse_args(sysargs)
    
    if args.version:
        return do_version(args)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    return cmds[args.command](args)

def do_version(args):
    from moanna.__meta__ import __version__
    print (__version__)

def do_predict(args):
    from moanna.main import moanna_predict

    proc = moanna_predict(
        args.input,
        args.summary,
        args.output,
        args.model,
        args.json,
    )

    if not (proc):
        print ("Moanna completed successfully")
    else:
        print ("Moanna completed with errors")
        
    raise SystemExit

def do_train(args):
    from moanna.main import moanna_train

    proc = moanna_train(
        args.input,
        args.label,
        args.output,
    )

    print (proc)

    raise SystemExit

def add_predict_args(parser, add_argument=True):

    parser.add_argument(
        "-i",
        "--input",
        help="Input features table",
        required=True,
    )

    parser.add_argument(
        "-s",
        "--summary",
        help="Output predictions summary table",
        required=True,
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Output prediction detailed table",
        required=True,
    )

    parser.add_argument(
        "-m",
        "--model",
        help="Pre-trained Model PyTorch State",
        required=True,
    )

    parser.add_argument(
        "-j",
        "--json",
        help="Pre-trained Model JSON parameters",
        required=True,
    )

    return parser

def add_train_args(parser, add_argument=True):

    parser.add_argument(
        "-i",
        "--input",
        help="Input features table",
        required=True,
    )

    parser.add_argument(
        "-l",
        "--label",
        help="Labels for training",
        required=True,
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Output table",
        required=True,
    )

    return parser