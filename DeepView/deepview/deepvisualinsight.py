
class MMS:
    def __init__(self, content_path, boundary_diff=1.5):
        '''
        This class contains the model management system (super DB) and provides
        several DVI user interface for dimension reduction and inverse projection function
        This class serves as a backend for DeepVisualInsight plugin.

        Parameters
        ----------
        '''
        self.visualization_models = None
        self.subject_models = None
        self.content_path = content_path
        self.training_data = None
        self.data_epoch_index = None
        self.testing_data = None
        self.boundary_diff = boundary_diff

    def load_content(self, content_path):
        pass

    def prepare_visualization_for_all(self):
        pass

    def visualize_model(self):
        pass

    def get_representation_layer(self):
        pass

    def individual_project(self):
        pass

    def batch_project(self):
        pass

    def individual_inverse_project(self):
        pass

    def batch_inverse_project(self):
        pass

    def get_incorrect_predictions(self):
        pass

    def _define_visualization_model(self):
        pass