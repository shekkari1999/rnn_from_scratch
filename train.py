import os
import torch

# Create a directory to store checkpoints
os.makedirs("checkpoints", exist_ok=True)

def train():
    train_loader, stoi, itos = prepare_data()

    # Define model
    hidden_dim = config.HIDDEN_DIM
    vocab_size = len(stoi)

    vocab_emb = torch.randn(vocab_size, config.EMBEDDING_DIM)  # Random embeddings (replace with GloVe)
    model = CustomRNN(vocab_emb, hidden_dim).to(config.DEVICE)

    # Loss function & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"🔥 Epoch {epoch+1}/{config.NUM_EPOCHS}", leave=True)

        for Xbatch, ybatch in train_loader:
            Xbatch, ybatch = Xbatch.to(config.DEVICE), ybatch.to(config.DEVICE)

            optimizer.zero_grad()
            output = model(Xbatch)

            # Compute loss
            loss = criterion(output.view(-1, vocab_size), ybatch.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Save checkpoint after each epoch
        checkpoint_path = f"checkpoints/rnn_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"✅ Model saved at {checkpoint_path}")

        print(f'🔥 Epoch {epoch + 1}: Loss {total_loss / len(train_loader):.4f}')

if __name__ == "__main__":
    train()