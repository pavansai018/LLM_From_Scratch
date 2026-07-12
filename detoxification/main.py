import torch
import torch.nn as nn
from utils import create_dataloader
from gpt_model import GPTModel
import config
from pathlib import Path

class GPTBatchProcessor:
    def __init__(self, model: nn.Module, device: torch.device, ignore_index: int = -100):
        self.model = model
        self.device = device

        self.loss_function = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def process(self, batch: list[dict[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)

        logits = self.model(input_ids)
        batch_size, sequence_length, vocab_size = logits.shape

        '''
        cross entropy loss expects:
        predictions: [num_tokens, vocab_size]
        targets: [num_tokens]
        '''

        logits = logits.reshape(batch_size * sequence_length, vocab_size)
        labels = labels.reshape(batch_size * sequence_length)

        loss = self.loss_function(logits, labels)

        return logits, loss

class GPTTrainer:
    def __init__(self, 
                 model: nn.Module, 
                 optimizer: torch.optim.Optimizer, 
                 device: torch.device,
                 ignore_index: int = -100,
                 max_grad_norm: float = 1.0,
    ):
        self.model: nn.Module = model
        self.optimizer: torch.optim.Optimizer = optimizer
        self.device: torch.device = device
        self.max_grad_norm: float = max_grad_norm
        self.loss_function: nn.CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def calculate_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)

        # [batch, seq_len, vocab_size]
        logits = self.model(input_ids)
        batch_size, sequence_length, vocab_size = logits.shape

        '''
        cross entropy loss expects:
        predictions: [num_tokens, vocab_size]
        targets: [num_tokens]
        '''

        logits = logits.reshape(batch_size * sequence_length, vocab_size)
        labels = labels.reshape(batch_size * sequence_length)

        loss = self.loss_function(logits, labels)
        return loss
    
    def train_step(self, batch: dict[str, torch.Tensor]) -> tuple[float, float]:
        # self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.calculate_loss(batch=batch)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)

        self.optimizer.step()

        return loss.item(), gradient_norm.item()
    
    def train_epoch(self, train_dataloader, epoch: int, log_interval: int = 100) -> float:
        self.model.train()
        total_loss = 0.0
        number_of_batches = 0
        for step, batch in enumerate(train_dataloader):
            loss, gradient_norm = self.train_step(batch)
            total_loss += loss
            number_of_batches += 1

            if step % log_interval == 0:
                print(
                f"epoch: {epoch:02d} | "
                f"step: {step:05d}/{len(train_dataloader):05d} | "
                f"loss: {loss:.4f} | "
                f"gradient norm: {gradient_norm:.4f}"
                )
        average_loss = total_loss / number_of_batches
        return average_loss
    
    def fit(self, train_dataloader, number_of_epochs: int):
        
        for epoch in range(1, number_of_epochs + 1):
            average_loss = self.train_epoch(train_dataloader=train_dataloader, epoch=epoch)
            print(
            f"epoch: {epoch:02d} completed | "
            f"average training loss: {average_loss:.4f}"
            )

        return average_loss
    
    def save_checkpoint(self, checkpoint_path: str, epoch: int, training_loss: float, model_config: dict) -> None:
        checkpoint_file = Path(checkpoint_path)
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "training_loss": training_loss,
            "model_config": model_config,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        torch.save(checkpoint, checkpoint_file)
        print(f'Checkpoint saved to: {checkpoint_file}')
    
def main():
    torch.manual_seed(123)
    train_dataloader = create_dataloader()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GPTModel(cfg=config.GPT2_SMALL_124M).to(device=device)

    '''
    
    Load pretrained gpt2 small model
    '''
    pretrained_path = '../models/gpt2_small_model_and_optimizer.pth'
    checkpoint = torch.load(pretrained_path, map_location='cpu',)
    print(checkpoint.keys())
    model.load_state_dict(checkpoint['model_state_dict'])

    # batch_processor = GPTBatchProcessor(model=model, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    trainer = GPTTrainer(model=model, optimizer=optimizer, device=device)
    num_epochs = 1
    final_loss = trainer.fit(train_dataloader=train_dataloader, number_of_epochs=num_epochs)
    trainer.save_checkpoint(
        checkpoint_path=f'checkpoints/detox_gpt2_epoch_{num_epochs}.pt',
        epoch=num_epochs,
        training_loss=final_loss,
        model_config=config.GPT2_SMALL_124M,
    )




if __name__ == '__main__':
    main()