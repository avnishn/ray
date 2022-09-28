class RLModule:
    """Base class for RLlib modules."""
    def __call__(self, batch, inference=False, **kwargs):
        if inference:
            return self.forward_inference(batch, **kwargs)
        return self.forward_train(batch, **kwargs)

    def forward_inference(self, batch, **kwargs):
        """Forward-pass during online sample collection
        Which could be either during training or evaluation based on explore parameter.
        """
        pass

    def forward_train(self, batch, **kwargs):
        """Forward-pass during computing loss function"""
        pass
