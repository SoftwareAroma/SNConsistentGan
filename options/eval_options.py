from .base_options import BaseOptions


class EvalOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument(
            '--results_dir', 
            type=str, 
            default='./results/', 
            help='saves results here.'
        )
        parser.add_argument(
            '--sn_gan', 
            type=int, default=1, 
            help="Cycle Gan with Spectral Normalization"
        )
        parser.add_argument(
            '--output_file_name', 
            type=str, 
            default="metrics_results", 
            help="Name of the file in which we should save the results"
        )
        parser.add_argument(
            '--metric', 
            type=str, 
            default='inception',  
            help='Which metric to evaluate'
        )
        parser.add_argument(
            '--testA', 
            type=str, 
            default='trainA',  
            help='Folder path for the input A image files'
        )
        parser.add_argument(
            '--testB', 
            type=str, 
            default='trainB',  
            help='Folder path for the input B image files'
        )
        parser.set_defaults(
            model='test'
        )
        # To avoid cropping, the loadSize should be the same as fineSize
        parser.set_defaults(
            loadSize=parser.get_default('fineSize')
        )
        self.isTrain = False
        return parser
