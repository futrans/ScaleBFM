import smplx
import torch


def create_smplx_model(body_model_path, gender):
    return smplx.create(
        body_model_path,
        "smplx",
        gender=str(gender),
        use_pca=False,
        num_betas=16,
    )


def expand_betas(betas, num_frames):
    if betas.shape[0] == num_frames:
        return betas
    if betas.shape[0] != 1:
        raise ValueError(
            f"betas batch must be 1 or match num_frames={num_frames}; "
            f"got {betas.shape[0]}"
        )
    return betas.expand(num_frames, -1)


class SmplxModelCache:

    def __init__(self, body_model_path):
        self.body_model_path = body_model_path
        self._models = {}

    def get(self, gender):
        gender = str(gender)
        if gender not in self._models:
            self._models[gender] = create_smplx_model(self.body_model_path, gender)
        return self._models[gender]


@torch.inference_mode()
def run_smplx_inference(body_model, **kwargs):
    return body_model(**kwargs)
