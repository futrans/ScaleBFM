import smplx
import torch

class SmplxModelCache:

    def __init__(self, body_model_path):
        self.body_model_path = body_model_path
        self._models = {}

    def get(self, gender):
        gender = str(gender)
        if gender not in self._models:
            self._models[gender] = smplx.create(
                self.body_model_path,
                "smplx",
                gender=gender,
                use_pca=False,
            )
        return self._models[gender]


@torch.inference_mode()
def run_smplx_inference(body_model, **kwargs):
    return body_model(**kwargs)
