class KnowledgeDistillationPipeline:
    def __init__(self, teacher_model, student_model, loss_fn, optimizer):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train(self, train_loader, num_epochs):
        for epoch in range(num_epochs):
            for inputs, targets in train_loader:
                # Forward pass
                teacher_outputs = self.teacher_model(inputs)
                student_outputs = self.student_model(inputs)

                # Compute loss
                loss = self.loss_fn(student_outputs, teacher_outputs)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def evaluate(self, test_loader):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                # Forward pass
                teacher_outputs = self.teacher_model(inputs)
                student_outputs = self.student_model(inputs)

                # Compute loss
                loss = self.loss_fn(student_outputs, teacher_outputs)

                # Update evaluation metrics
                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(student_outputs.data, 1)
                total_correct += (predicted == targets).sum().item()
                total_samples += inputs.size(0)

        average_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return average_loss, accuracy
