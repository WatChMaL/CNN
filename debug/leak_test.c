#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

int main()
{
	unsigned long size = 2000000000;
	unsigned long i;
	char *p;
	int fd;
	char sum;

	fd = open("data", O_RDONLY);
	p = (char*)mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);

				        sum = 0;
	for (i = 0; i < size; ++i) {
		sum += *(p + i);
	}
	munmap(p, size);
	close(fd);

	return 0;
}
