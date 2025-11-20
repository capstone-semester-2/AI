# MLP Adapter for DeepSpeech2 - Personalized Speech Recognition

## Overview

This implementation adds MLP (Multi-Layer Perceptron) adapter functionality to DeepSpeech2, enabling personalized speech recognition for individual users. The adapter allows users to fine-tune the model for their specific voice characteristics without modifying the pre-trained base model.

## Key Features

✅ **Non-destructive fine-tuning**: Only the MLP adapter is trained, base model parameters remain frozen  
✅ **Separate storage**: Adapters are saved as individual `.pt` files independent from the base model  
✅ **Efficient training**: Only adapter parameters are updated during training  
✅ **Easy integration**: Simple API for loading/saving adapters  
✅ **Multiple adapters**: Train and manage different adapters for different users  

## Architecture

### MLP Adapter Structure

```
Input (RNN output) → Linear(dim → hidden_dim[0]) → ReLU → Dropout
                   → Linear(hidden_dim[0] → hidden_dim[1]) → ReLU → Dropout
                   → Linear(hidden_dim[1] → num_classes)
                   → Output (predicted tokens)
```

### Training Configuration

- **Base Model**: DeepSpeech2 encoder + RNN layers (FROZEN during adapter training)
- **Adapter**: 2-3 layer MLP appended to base model (TRAINABLE)
- **Output**: Adapter generates personalized predictions

## File Structure

```
kospeech/
├── models/
│   ├── adapter.py              # MLPAdapter class definition
│   ├── adapter_manager.py      # Save/load utilities for adapters
│   ├── deepspeech2/
│   │   └── model.py           # Updated DeepSpeech2 with adapter support
│   └── model.py               # Base model classes
├── trainer/
│   ├── adapter_trainer.py     # AdapterTrainer for adapter-only training
│   └── __init__.py            # AdapterTrainConfig added
└── model_builder.py           # build_deepspeech2 updated for adapter support
```

## Usage

### 1. Basic Setup - Create Model with Adapter

```python
from kospeech.models import DeepSpeech2
import torch

# Create DeepSpeech2 model with adapter
model = DeepSpeech2(
    input_dim=256,
    num_classes=2000,
    num_rnn_layers=5,
    rnn_hidden_dim=512,
    use_adapter=True,                          # Enable adapter
    adapter_hidden_dims=[512, 256],           # 2 hidden layers
    device=torch.device('cuda')
)

# Freeze base model, adapter is trainable by default
model.freeze_base_model()
```

### 2. Train Adapter

```python
from kospeech.trainer import AdapterTrainer
from omegaconf import OmegaConf

# Load configuration
config = OmegaConf.load('configs/train.yaml')

# Create trainer
trainer = AdapterTrainer(
    optimizer=optimizer,
    criterion=criterion,
    trainset_list=trainset_list,
    validset=validset,
    num_workers=config.train.num_workers,
    device=device,
    print_every=config.train.print_every,
    save_result_every=config.train.save_result_every,
    checkpoint_every=config.train.checkpoint_every,
    vocab=vocab,
    adapter_save_dir='./adapters'
)

# Train adapter
model = trainer.train(
    model=model,
    batch_size=config.train.batch_size,
    epoch_time_step=epoch_time_step,
    num_epochs=config.train.num_epochs,
    adapter_name='user_john'  # Unique adapter name
)
```

### 3. Save and Load Adapters

```python
from kospeech.models import AdapterManager

manager = AdapterManager()

# Save adapter
manager.save_adapter(
    model=model,
    save_path='./adapters',
    adapter_name='user_john'
)
# Saves to: ./adapters/user_john_adapter.pt

# Load adapter
manager.load_adapter(
    model=model,
    adapter_path='./adapters/user_john_adapter.pt'
)

# Get adapter info without loading
info = manager.get_adapter_info('./adapters/user_john_adapter.pt')
# Returns: {'name', 'input_dim', 'hidden_dims', 'output_dim'}

# Save/load both model and adapter
manager.save_model_with_adapter(
    model=model,
    model_path='./models/deepspeech2.pt',
    adapter_path='./adapters',
    adapter_name='user_john'
)

manager.load_model_with_adapter(
    model=model,
    model_path='./models/deepspeech2.pt',
    adapter_path='./adapters/user_john_adapter.pt'
)
```

### 4. Forward Pass with Adapter

```python
# When use_adapter=True, forward returns 3 values
base_output, output_lengths, adapter_output = model(inputs, input_lengths)

# base_output: predictions from original FC layer (not used)
# adapter_output: personalized predictions from adapter (use for loss/recognition)
# output_lengths: sequence lengths
```

### 5. Check Parameter Counts

```python
# Get detailed parameter information
param_info = model.module.count_parameters(trainable_only=False)
print(f"Total parameters: {param_info['total']}")
print(f"Base model parameters: {param_info['base']}")
print(f"Adapter parameters: {param_info['adapter']}")

# Only trainable parameters
trainable_info = model.module.count_parameters(trainable_only=True)
```

## Training Workflow

```
1. Load pre-trained DeepSpeech2 model
                    ↓
2. Attach MLP Adapter (input_dim matches RNN output dim)
                    ↓
3. Freeze base model parameters
                    ↓
4. Create AdapterTrainer with adapter-specific learning rate
                    ↓
5. Train on user-specific data (small dataset OK)
                    ↓
6. Adapter learns personalized patterns while base model stays fixed
                    ↓
7. Save adapter .pt file separately
                    ↓
8. Load adapter for inference or further training
```

## Example Configuration

```yaml
# configs/adapter_train.yaml
train:
  batch_size: 32
  num_epochs: 10
  num_workers: 4
  use_cuda: true
  output_unit: "character"
  
  # Optimizer settings for adapter
  init_lr: 1e-04
  final_lr: 1e-05
  peak_lr: 1e-04
  warmup_steps: 500
  lr_scheduler: 'tri_stage_lr_scheduler'

model:
  architecture: "deepspeech2"
  hidden_dim: 512
  num_encoder_layers: 5
  dropout: 0.1
  activation: "hardtanh"

# Adapter settings
adapter:
  hidden_dims: [512, 256]
  name: "user_john"
  save_dir: "./adapters"
```

## Adapter Storage Format

Each adapter is saved as a `.pt` file containing:

```python
{
    'adapter_state_dict': {...},      # Adapter weights and biases
    'input_dim': 1024,                # RNN output dimension
    'hidden_dims': [512, 256],        # Hidden layer dimensions
    'output_dim': 2000,               # Number of output classes
    'adapter_name': 'user_john',      # Adapter identifier
}
```

## Performance Characteristics

| Aspect | Benefit |
|--------|---------|
| **Training Speed** | Faster than full model fine-tuning (fewer parameters) |
| **Adaptation Efficiency** | Works well with small user-specific datasets |
| **Memory Usage** | Lower than full model training |
| **Storage** | Minimal (only adapter .pt file) |
| **Inference Speed** | Same as normal DeepSpeech2 (adapter is small) |

## Advantages

1. **User Personalization**: Each user gets their own adapter for voice-specific adaptation
2. **Base Model Protection**: Original model weights never change
3. **Easy Distribution**: Only share adapter files between users
4. **Flexible Training**: Train adapters with limited data per user
5. **Multiple Adapters**: Maintain different adapters for different speakers

## Advanced Usage

### Unfreeze Base Model for Full Fine-tuning

```python
model.module.unfreeze_base_model()  # Now all parameters are trainable
```

### Transfer Learning Between Adapters

```python
# Load one adapter's weights
manager.load_adapter(model, 'adapters/user_john_adapter.pt')

# Continue training for another user
model = trainer.train(
    model=model,
    ...,
    adapter_name='user_jane'  # Saves as different adapter
)
```

### Inspect Adapter Architecture

```python
print(model.module.adapter)
# Output: MLPAdapter structure with layer information
```

## Troubleshooting

### Adapter not updating during training?
- Check that `model.module.use_adapter=True`
- Verify `model.module.adapter` is not None
- Check that base model is frozen: `model.module.freeze_base_model()`

### High loss during adapter training?
- Start with higher learning rate (e.g., 1e-4)
- Increase adapter hidden dimensions
- Ensure proper data preprocessing
- Check that adapter input dimensions match RNN output

### Adapter file won't load?
- Verify file path is correct
- Ensure model has adapter initialized with same architecture
- Check adapter configuration (input_dim, hidden_dims, output_dim match)

## References

- DeepSpeech2 Paper: https://arxiv.org/abs/1512.02595
- Adapter concept: Parameter-efficient transfer learning
- Implementation: PyTorch

## License

Apache License 2.0 - See LICENSE file for details

## Authors

- Adapter implementation for personalized speech recognition (2025)
- Based on KoSpeech framework by Soohwan Kim
