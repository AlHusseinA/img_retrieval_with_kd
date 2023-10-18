from src.dataloaders import getCub2011Loaders






root = "/media/alabutaleb/09d46f11-3ed1-40ce-9868-932a0133f8bb/data/cub200/"
num_workers=10

trainloader, testloader = getCub2011Loaders(root, num_workers=10)

print("#"*30)
print(trainloader)
print(testloader)
print("#"*30)



