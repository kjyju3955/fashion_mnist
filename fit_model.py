import torch
from torch.autograd import Variable
import data_pro
import make_model


def fit_model(device):
    model, error, optimizer = make_model.make_model(device)
    train_loader, test_loader = data_pro.get_data()

    num_epochs = 5
    count = 0

    loss_list = []
    iteration_list = []
    accuracy_list = []

    predictions_list = []
    labels_list = []

    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            train = Variable(images.view(100, 1, 28, 28))
            labels = Variable(labels)

            outputs = model(train)
            loss = error(outputs, labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            count += 1

            if not (count % 50):
                total = 0
                correct = 0
                for image, label in test_loader:
                    images, labels = image.to(device), label.to(device)
                    labels_list.append(labels)

                    test = Variable(images.view(100, 1, 28, 28))

                    outputs = model(test)

                    predictions = torch.max(outputs, 1)[1].to(device)
                    predictions_list.append(predictions)
                    correct += (predictions == labels).sum()

                    total += len(labels)

                accuracy = correct * 100 / total
                loss_list.append(loss.data)
                iteration_list.append(count)
                accuracy_list.append(accuracy)

            if not (count % 500):
                print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))
