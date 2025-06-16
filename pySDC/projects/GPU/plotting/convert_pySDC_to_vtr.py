def convert_to_vtr(path, outpath, varNames):  # pragma: no cover
    from pySDC.helpers.fieldsIO import FieldsIO

    FieldsIO.fromFile(path).toVTR(baseName=outpath, varNames=varNames)


if __name__ == '__main__':  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert pySDC file to vtr files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--path", help="path to the .pySDC file")
    parser.add_argument("--outpath", help="path to the output directory")
    parser.add_argument("--problem", help="problem")
    args = parser.parse_args()

    if args.problem == 'RBC3D':
        varNames = ['u', 'v', 'w', 'T', 'p']
    else:
        raise NotImplementedError(f'Don\'t know what variables are stored in run of {args.problem}!')

    convert_to_vtr(args.path, args.outpath, varNames)
