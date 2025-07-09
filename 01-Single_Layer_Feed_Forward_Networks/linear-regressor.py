'''
    Linear Regressor for the equation "y = 2x + 1".
'''

# Training data
x_vals = [0,1,2,3,4]
y_vals = [1,3,5,7,9]

# Initial weight and bias
weight, bias = 0.0, 0.0
learning_rate = 0.01 # Learning Rate

for epoch in range(1000):
    total_loss = 0
    for x, y in zip(x_vals, y_vals):
        y_pred = weight * x + bias
        error = y - y_pred
        weight += learning_rate * error * x
        bias += learning_rate * error
        total_loss += error ** 2
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {total_loss}')

print(f'Learned weight: {weight}, bias: {bias}')