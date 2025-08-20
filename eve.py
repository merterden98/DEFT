import cli


def main():
    parser = cli.parser
    args = parser.parse_args()
    args.cli.run()


if __name__ == '__main__':
    main()

