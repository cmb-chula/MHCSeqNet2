class LangRunconfig:
    LABEL = "Run config"
    HELP = "Program parameter configs"

    LOG_DIR_LABEL = "Log unknow"
    LOG_DIR_TITLE = "Select unknown log save location"
    LOG_DIR_HELP = "Log the unknown to a file.\nAt"

    IGNORE_UNK_LABEL = "Ignore unknow"
    IGNORE_UNK_HELP = "If not check, when encounter an unknown peptied or allele, it will throw error"

    ENSEMBLE_LABEL = "Use ensemble"
    ENSEMBLE_HELP = "Use average of 5 k-fold model\'s prediction.\nBetter accuracy, slower."

    GPU_LABEL = "Select GPU"
    GPU_HELP = "Select processing hardware - CPU or GPU\nIt's grayed out when it could not detect any gpu device"
    NO_GPU = "CPU"

    ADD_ALLELE = "Use custom allele sequence path"
    ADD_ALLELE_HELP = ("Add allele amino sequence infomation (deal with unknown)\n\n"
                       "Normally, the tool will search for allele information through all .yaml files in ascending order\n"
                       "located at `resources/allele_mapper/0_allele_mapper.yaml` each line is in this format\n"
                       "`'HLA-A*01:01': 'MAVMAPRTLLLLL...SDVSLTACKV'`\n"
                       "You can either create a new file e.g. `resources/allele_mapper/1_myallele.yaml` or\n"
                       "use the text right side to select the new folder to be used.\n\n"
                       "***You can read more about how to prepare the sequence in the readme")  # TODO: add github link
    ADD_ALLELE_BROWS_TEXT = "Select Allele Sequence Folder"

    OUTPUT_LABEL = "Output file"
    OUTPUT_HELP = "select file where to save output.csv"

    RANK_EL_LABEL = "Rank EL"
    RANK_EL_HELP = ("Whether or not to calculate Rank EL\n\n"
                    "Slow and use a lots of memory.\nYou'll have to use a threshold between [0, 1] to determine the strong and weak binder - normally, 0.005, 0.02\n\n"
                    "%Rank_EL: Rank of the predicted binding score compared to a set of random natural peptides.\n"
                    "This measure is not affected by inherent bias of certain molecules towards higher or lower mean predicted affinities.\n"
                    "Strong binders are defined as having rank<0.005, and weak binders with rank<0.02. We advise to select candidate binders based on Rank EL rather than Score")


class LangModeCSV:
    LABEL = 'CSV'
    DEFAULT_BROWS_TEXT = "Select CSV file"
    HELP = ("A csv/tsv file which can either be one of the following\n"
            "1. File containing column name 'peptide', 'allele'.\n"
            "2. File with no column name but first and second column is peptide and allele respectively")

    PREVIEW_TITLE = "Preview CSV file"
    PREVIEW_HELP = "If you notice 2 rows being text,\nit means that column name is not match and thus can't be identify.\nplease edit the csv file"

    VALIDATE_ERROR_SUM_TEXT = "CSV selection failed"
    VALIDATE_ERROR_INFO = "Column Name is not supported"
    VALIDATE_ERROR_DETAIL1 = "The details are as follows:\n"
    VALIDATE_ERROR_DETAIL2 = "It can either not has the column name row, then it will use the first column as peptide and the next as allele, or, it can use one of the following\npeptide, Peptide, PEPTIDE, allele, Allele, ALLELE"


class LangModeCross:
    LABEL = 'Peptide X Allele'
    PEPTIDE_BROWS_TEXT = "Select Peptide file"
    PEPTIDE_HELP = ("A txt or csv/tsv file which can either be one of the following\n"
                    "• A file with one peptide per line\n"
                    "• A csv/tsv file containing column name 'peptide'")
    PREVIEW_PEPTIDE_TITLE = "Preview peptide"
    PREVIEW_PEPTIDE_HELP = "List of peptide"

    ALLELE_BROWS_TEXT = "Select allele file"
    ALLELE_HELP = ("A txt or csv/tsv file which can either be one of the following\n"
                   "• A file with one allele per line i.e. HLA-A*01:01\n"
                   "• A csv/tsv file containing column name 'allele'")
    PREVIEW_ALLELE_TITLE = "Preview allele"
    PREVIEW_ALLELE_HELP = "List of allele"


class Lang:
    MODE = ("Select the mode to run, the tool will execute based on current selection\n"
            "• csv mode allow you to choose a csv/tsv file which must contain the column for peptide and allele\n"
            "• cross mode allow you to choose two files one containing peptides, and the other containing alleles which will be crossed together")
    RUNCONFIG = LangRunconfig()
    CSV = LangModeCSV()
    CROSS = LangModeCross()
    # TODO: error translation
