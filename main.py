#This is the entrance mwthod for the framework.
import sys

import preload
import preprocess


def main(genotype_file):
    preprocess.main(genotype_file)


if __name__ == '__main__':
    main(sys.argv[1])

