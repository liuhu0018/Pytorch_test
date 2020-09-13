from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
# y = x
for i in range(10):
    writer.add_scalar('y = 3x', 3*i, i)
# writer.add_image()
# writer.add_scalar()

writer.close()
