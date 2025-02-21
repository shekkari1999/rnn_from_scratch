import logging
import os
import torch

# Create logs directory if not exists
os.makedirs("logs", exist_ok=True)

# Configure Logging
logging.basicConfig(
    filename="logs/training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def log_training(epoch, total_loss, train_loader):
    """Logs training loss per epoch"""
    avg_loss = total_loss / len(train_loader)
    log_msg = f"🔥 Epoch {epoch+1}: Loss {avg_loss:.4f}"
    print(log_msg)  # Print to console
    logging.info(log_msg)  # Log to file

def save_model_weights(model, epoch):
    """Saves model weights after every epoch"""
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pt")
    logging.info(f"✅ Saved model weights for epoch {epoch+1}")