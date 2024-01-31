#include "slic.h"

//VARIABILI GLOBALI
int N = 0;
int K = 0;
int sudd = 0;
int S = 0;
int ngbSize = 0;
int m = 10; //compactness constant
int height;
int width;
center *ck;
pixel **pi;

float cpumil;

void cpuSLIC(const char *input , int sud)
{
  bitmap_image image(input);
  if (!image)
  {
    printf("Error - Failed to open input_Image\n");
    return;
  }
  height = image.height();
  width = image.width();
  N = width * height;
  sudd = sud;
  float step = round(((float)width) / sudd);
  float rows = round(((float)height) / step);
  K = (sudd) * (rows);
  S = sqrt(((float)N) / K);
  ngbSize = 2 * S;
  printf("CPUSLIC\n\nwidth: %d, height: %d\n", width, height);
  printf("N: %d, K: %d, S: %d\n\n", N, K, S);

  //INIZIALIZZAZIONE CENTRI E PIXEL
  ck = (center *)malloc(sizeof(center) * K);
  pi = (pixel **)malloc(sizeof(pixel*) * height);
  for (int i = 0; i < height; i++)
  {
    pi[i] = (pixel *)malloc(sizeof(pixel) * width);
  }
  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < width; x++)
    {
      rgb_t colour;
      image.get_pixel(x, y, colour);
      pi[y][x].r = colour.red;
      pi[y][x].g = colour.green;
      pi[y][x].b = colour.blue;
      pi[y][x].x = x;
      pi[y][x].y = y;
      pi[y][x].cluster = -1;
      pi[y][x].distance = INT_MAX;
    }
  }
  int nx = 0;
  int ny = 0;
  int xs = S / 2;
  int ys = S / 2;
  rgb_t colour;
  for (int i = 0; i < K; i++)
  {
    lowestGradient(xs, ys, &image, &nx, &ny);
    image.get_pixel(nx, ny, colour);
    ck[i].x = nx;
    ck[i].y = ny;
    ck[i].r = colour.red;
    ck[i].g = colour.green;
    ck[i].b = colour.blue;
    if ((xs + S) > (width))
    {
      ys = (ys + S);
      xs = S / 2;
    }
    else
    {
      xs = (xs + S);
    }
  }
  //Creazione di eventi per performance metric
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  //ASSIGNMENT DEI PIXEL CON I CENTER PIU' VICINI
  int cx;
  int cy;
  int startx;
  int starty;
  double tempD;
  int iteration = 0;
  long counters[K];
  counterCenter centerAcc[K];
  do
  {
    for (int i = 0; i < K; i++)
    {
      cx = ck[i].x;
      cy = ck[i].y;
      startx = cx - S;
      starty = cy - S;
      for (int y = starty; y < starty + ngbSize; y++)
      {
        for (int x = startx; x < startx + ngbSize; x++)
        {
          if (x >= 0 && x < width && y >= 0 && y < height)
          {
            tempD = ds(&ck[i], pi[y][x].r, pi[y][x].g, pi[y][x].b, x, y);
            if (tempD < pi[y][x].distance)
            {
              pi[y][x].distance = tempD;
              pi[y][x].cluster = i;
            }
          }
        }
      }
    }
    //reset counters
    for (int i = 0; i < K; i++)
    {
      counters[i] = 0;
      centerAcc[i].r = 0;
      centerAcc[i].g = 0;
      centerAcc[i].b = 0;
      centerAcc[i].x = 0;
      centerAcc[i].y = 0;
    }
    //compute new cluster centers con media iterativa
    for (int y = 0; y < height; y++)
    {
      for (int x = 0; x < width; x++)
      {
        if (counters[pi[y][x].cluster] == 0)
        {
          centerAcc[pi[y][x].cluster].r += pi[y][x].r;
          centerAcc[pi[y][x].cluster].g += pi[y][x].g;
          centerAcc[pi[y][x].cluster].b += pi[y][x].b;
          centerAcc[pi[y][x].cluster].x += pi[y][x].x;
          centerAcc[pi[y][x].cluster].y += pi[y][x].y;
          counters[pi[y][x].cluster]++;
        }
        else
        {
          centerAcc[pi[y][x].cluster].r += (pi[y][x].r - centerAcc[pi[y][x].cluster].r) * (1.0 / (counters[pi[y][x].cluster] + 1));
          centerAcc[pi[y][x].cluster].g += (pi[y][x].g - centerAcc[pi[y][x].cluster].g) * (1.0 / (counters[pi[y][x].cluster] + 1));
          centerAcc[pi[y][x].cluster].b += (pi[y][x].b - centerAcc[pi[y][x].cluster].b) * (1.0 / (counters[pi[y][x].cluster] + 1));
          centerAcc[pi[y][x].cluster].x += (pi[y][x].x - centerAcc[pi[y][x].cluster].x) * (1.0 / (counters[pi[y][x].cluster] + 1));
          centerAcc[pi[y][x].cluster].y += (pi[y][x].y - centerAcc[pi[y][x].cluster].y) * (1.0 / (counters[pi[y][x].cluster] + 1));
          counters[pi[y][x].cluster]++;
        }
      }
    }
    for (int i = 0; i < K; i++)
    {
      ck[i].r = (pel)centerAcc[i].r;
      ck[i].g = (pel)centerAcc[i].g;
      ck[i].b = (pel)centerAcc[i].b;
      ck[i].x = (int)centerAcc[i].x;
      ck[i].y = (int)centerAcc[i].y;
    }
    iteration++;
  } while (iteration < 10);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cpumil, start, stop);
  printf("CPUSLIC milliseconds: %f\n", cpumil);
}

void outputSlic(int bordi, int centri,const char *output)
{
  //SCRITTURA BMP usando i PIXEL struct
  bitmap_image result(width, height);
  result.clear();
  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
      result.set_pixel(x, y, pi[y][x].r, pi[y][x].g, pi[y][x].b);
    }
  }
  //Disegno i bordi dei cluster
  if (bordi)
  {
    int cluster = 0;
    int upp = 0;
    int lowp = 0;
    int rightp = 0;
    int leftp = 0;
    for (int y = 0; y < height; ++y)
    {
      for (int x = 0; x < width; ++x)
      {
        cluster = pi[y][x].cluster;
        upp = y - 1;
        lowp = y + 1;
        rightp = x + 1;
        leftp = x - 1;
        if (upp >= 0 && lowp < height && rightp < width && leftp >= 0)
        {
          if (pi[upp][x].cluster != cluster || pi[lowp][x].cluster != cluster || pi[y][leftp].cluster != cluster || pi[y][rightp].cluster != cluster)
            result.set_pixel(x, y, 51, 20, 203);
        }
      }
    }
  }
  else
  {
    //Disegno tutti i pixel con il colore del loro cluster center
    for (int y = 0; y < height; ++y)
    {
      for (int x = 0; x < width; ++x)
      {
        result.set_pixel(x, y, ck[pi[y][x].cluster].r, ck[pi[y][x].cluster].g, ck[pi[y][x].cluster].b);
      }
    }
  }
  // Disegno i cluster centers
  if (centri)
  {
    for (int i = 0; i < K; i++)
    {
      result.set_pixel(ck[i].x, ck[i].y, 111, 0, 255);
    }
  }
  result.save_image(output);
}

double ds(center *c, pel ri, pel gi, pel bi, int xi, int yi)
{
  double rs = pow(c->r - ri, 2);
  double gs = pow(c->g - gi, 2);
  double bs = pow(c->b - bi, 2);
  double dRGB = sqrt(rs + gs + bs);

  double xs = pow(c->x - xi, 2);
  double ys = pow(c->y - yi, 2);
  double dxy = sqrt(xs + ys);

  return dRGB + (m / S) * dxy;
}

void lowestGradient(int cx, int cy, bitmap_image *image, int *nx, int *ny)
{
  double min_grad = DBL_MAX;
  double red_g = 0;
  double green_g = 0;
  double blue_g = 0;
  double magn = 0;
  int local_minx = 0;
  int local_miny = 0;
  rgb_t colour1;
  rgb_t colour2;
  rgb_t colour3;
  rgb_t colour4;
  int width = image->width();
  int height = image->height();
  for (int x = cx - 1; x < cx + 2; x++)
  {
    for (int y = cy - 1; y < cy + 2; y++)
    {
      if (x >= 0 && x < width && y >= 0 && y < height)
      {
        image->get_pixel(x + 1, y, colour1);
        image->get_pixel(x - 1, y, colour2);
        image->get_pixel(x, y + 1, colour3);
        image->get_pixel(x, y - 1, colour4);
        red_g = sqrt(pow(colour1.red - colour2.red, 2)) + sqrt(pow(colour3.red - colour4.red, 2));
        green_g = sqrt(pow(colour1.green - colour2.green, 2)) + sqrt(pow(colour3.green - colour4.green, 2));
        blue_g = sqrt(pow(colour1.blue - colour2.blue, 2)) + sqrt(pow(colour3.blue - colour4.blue, 2));
        magn = sqrt(pow(red_g, 2) + pow(green_g, 2) + pow(blue_g, 2));
        if (min_grad > magn)
        {
          min_grad = magn;
          local_minx = x;
          local_miny = y;
        }
      }
    }
  }
  *nx = local_minx;
  *ny = local_miny;
}

float getCpuMil() {
  return cpumil;
}