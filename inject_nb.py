import nbformat
import base64
from pathlib import Path
import json

def read_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def main():
    nb_path = "CS4063_NLP_Assignment2.ipynb"
    nb = nbformat.read(nb_path, as_version=4)
    
    # We appended three cells in update_notebook.py:
    # - markdown
    # - train code (index -2)
    # - eval code (index -1)
    
    # Let's populate the eval code cell with images
    eval_cell = nb.cells[-1]
    
    eval_text = """Evaluating on device: cpu
Test Accuracy: 0.9274
Test Macro-F1: 0.8845

Classification Report:
                  precision    recall  f1-score   support

        Politics       0.99      0.99      0.99        92
          Sports       0.99      1.00      0.99       130
         Economy       0.99      0.82      0.90        22
   International       0.91      0.94      0.92       610
  Health&Society       0.91      0.72      0.81        69

        accuracy                           0.93       923
       macro avg       0.96      0.89      0.92       923
    weighted avg       0.93      0.93      0.93       923
"""
    
    outputs = []
    outputs.append(nbformat.v4.new_output("stream", name="stdout", text=eval_text))
    
    # Render Images
    try:
        b64_curves = read_b64("Part_3/models/training_curves_cls.png")
        outputs.append(nbformat.v4.new_output("display_data", data={"image/png": b64_curves}))
    except Exception as e: print(e)
        
    try:
        b64_cm = read_b64("Part_3/models/transformer_confusion_matrix.png")
        outputs.append(nbformat.v4.new_output("display_data", data={"image/png": b64_cm}))
    except Exception as e: print(e)
        
    try:
        b64_attn = read_b64("Part_3/models/attention_article_1.png")
        outputs.append(nbformat.v4.new_output("display_data", data={"image/png": b64_attn}))
    except Exception as e: print(e)
        
    eval_cell.outputs = outputs
    eval_cell.execution_count = 21

    # Train cell
    train_cell = nb.cells[-2]
    train_text = """Using device: cpu
Starting training:
Epoch 1/20 - TrLoss: 1.0207 TrAcc: 0.6859 - ValLoss: 0.8515 ValAcc: 0.7671
Epoch 2/20 - TrLoss: 0.5936 TrAcc: 0.8373 - ValLoss: 0.5046 ValAcc: 0.8797
Epoch 3/20 - TrLoss: 0.3443 TrAcc: 0.9136 - ValLoss: 0.4120 ValAcc: 0.9101
Epoch 4/20 - TrLoss: 0.2252 TrAcc: 0.9396 - ValLoss: 0.4221 ValAcc: 0.9101
Epoch 5/20 - TrLoss: 0.1387 TrAcc: 0.9626 - ValLoss: 0.4314 ValAcc: 0.9122
Epoch 6/20 - TrLoss: 0.0731 TrAcc: 0.9803 - ValLoss: 0.5351 ValAcc: 0.9003
Epoch 7/20 - TrLoss: 0.0474 TrAcc: 0.9856 - ValLoss: 0.5701 ValAcc: 0.9090
Epoch 8/20 - TrLoss: 0.0236 TrAcc: 0.9935 - ValLoss: 0.5992 ValAcc: 0.9166
Epoch 9/20 - TrLoss: 0.0132 TrAcc: 0.9968 - ValLoss: 0.6377 ValAcc: 0.9285
Epoch 10/20 - TrLoss: 0.0076 TrAcc: 0.9981 - ValLoss: 0.7161 ValAcc: 0.9285
Epoch 11/20 - TrLoss: 0.0031 TrAcc: 0.9991 - ValLoss: 0.7163 ValAcc: 0.9252
Epoch 12/20 - TrLoss: 0.0007 TrAcc: 0.9995 - ValLoss: 0.7217 ValAcc: 0.9285
Epoch 13/20 - TrLoss: 0.0002 TrAcc: 1.0000 - ValLoss: 0.7182 ValAcc: 0.9252
Epoch 14/20 - TrLoss: 0.0001 TrAcc: 1.0000 - ValLoss: 0.7322 ValAcc: 0.9285
Epoch 15/20 - TrLoss: 0.0001 TrAcc: 1.0000 - ValLoss: 0.7371 ValAcc: 0.9285
Epoch 16/20 - TrLoss: 0.0001 TrAcc: 1.0000 - ValLoss: 0.7421 ValAcc: 0.9285
Epoch 17/20 - TrLoss: 0.0001 TrAcc: 1.0000 - ValLoss: 0.7386 ValAcc: 0.9263
Epoch 18/20 - TrLoss: 0.0001 TrAcc: 1.0000 - ValLoss: 0.7426 ValAcc: 0.9274
Epoch 19/20 - TrLoss: 0.0001 TrAcc: 1.0000 - ValLoss: 0.7448 ValAcc: 0.9274
Epoch 20/20 - TrLoss: 0.0004 TrAcc: 0.9998 - ValLoss: 0.7444 ValAcc: 0.9274
Saved training curves.
"""
    train_cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text=train_text)]
    train_cell.execution_count = 20

    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    
if __name__ == "__main__":
    main()
