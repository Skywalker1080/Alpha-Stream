import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.exception.exceptions import PrismException
from src.config.pipeline_config import Config
from logger.logger import get_logger
from torch.optim import Adam

logger = get_logger()

def fit_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int = 10, device: str = Config().device, lr: float = 1e-3) -> nn.Module:
    try:
        model.to(device)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-6)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)
        best_val_loss = float('inf')
        patience, counter = 5, 0

        for epoch in range(1, num_epochs + 1):
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                X, Y = batch
                X, Y = X.to(device), Y.to(device)
                optimizer.zero_grad()
                pred = model(X)
                loss = criterion(pred, Y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total_loss += loss.item()
            avg_train_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {ep}/{epochs} - Train Loss: {avg_train_loss:.5f}")

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    X, Y = batch
                    X, Y = X.to(device), Y.to(device)
                    pred = model(X)
                    val_loss += criterion(pred, Y).item()
            avg_val_loss = val_loss / len(val_loader)
            logger.info(f"Epoch {ep}/{epochs} - Validation Loss: {avg_val_loss:.5f}")

            # Mflow tracking later

            scheduler.step(avg_val_loss)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    logger.info("Early stopping triggered")
                    break

        return model
    except PrismException as e:
        logger.exception(f"Failed to train model: {e}")
        raise PrismException(e)
                    
                
        

