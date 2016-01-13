/*
 *	Copyright (c) 2013, Nenad Markus
 *	All rights reserved.
 *
 *	This is an implementation of the algorithm described in the following paper:
 *		N. Markus, M. Frljak, I. S. Pandzic, J. Ahlberg and R. Forchheimer,
 *		Object Detection with Pixel Intensity Comparisons Organized in Decision Trees,
 *		http://arxiv.org/abs/1305.4537
 *
 *	Redistribution and use of this program as source code or in binary form, with or without modifications, are permitted provided that the following conditions are met:
 *		1. Redistributions may not be sold, nor may they be used in a commercial product or activity without prior permission from the copyright holder (contact him at nenad.markus@fer.hr).
 *		2. Redistributions may not be used for military purposes.
 *		3. Any published work which utilizes this program shall include the reference to the paper available at http://arxiv.org/abs/1305.4537
 *		4. Redistributions must retain the above copyright notice and the reference to the algorithm on which the implementation is based on, this list of conditions and the following disclaimer.
 *
 *	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 *	IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <omp.h>

#include <string>
#include <vector>

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <malloc.h>
#include <stdint.h>

struct Detection
{
	int x;
	int y;
	int w;
	int h;

	int image_idx;
	int obj_class;
	float score;
};

#define MAX_N 2000000
struct Dataset
{
	Dataset()
	{
		background.reserve(MAX_N);
		objects.reserve(MAX_N);
	}

	std::vector<uint8_t> ppixels[MAX_N];
	int pdims[MAX_N][2]; // (nrows, ncols)

	// indexes of images in ppixels
	// negatives should be sampled from these images
	std::vector<int> background;

	// positive samples
	// image_idx is related to ppixels
	std::vector<Detection> objects;

	// hard negatives (whole image is a negative sample)
	std::vector<int> negatives;
};
static Dataset dataset;
static int cur_stage = 0;

struct Cascade
{
	Cascade() :
		tsr(1.0f),
		tsc(1.0f),
		tdepth(5),
		ntrees(0)
	{}

	bool load_from_file(const char* path)
	{
		FILE* file = fopen(path, "rb");
		if (!file)
			return false;

		fread(&tsr, sizeof(float), 1, file);
		fread(&tsc, sizeof(float), 1, file);
		fread(&tdepth, sizeof(int), 1, file);

		fread(&ntrees, sizeof(int), 1, file);

		for (int i = 0; i < ntrees; ++i)
		{
			fread(&tcodes[i][0], sizeof(int32_t), (1<<tdepth)-1, file);
			fread(&luts[i][0], sizeof(float), 1<<tdepth, file);
			fread(&thresholds[i], sizeof(float), 1, file);
		}

		fclose(file);
		return true;
	}

	bool save_to_file(const char* path)
	{
		printf("* saving cascade...");
		fflush(stdout);
		FILE* file = fopen(path, "wb");
		if (!file)
			return false;

		fwrite(&tsr, sizeof(float), 1, file);
		fwrite(&tsc, sizeof(float), 1, file);
		fwrite(&tdepth, sizeof(int), 1, file);

		fwrite(&ntrees, sizeof(int), 1, file);
		for (int i = 0; i < ntrees; ++i)
		{
			fwrite(&tcodes[i][0], sizeof(int32_t), (1<<tdepth)-1, file);
			fwrite(&luts[i][0], sizeof(float), 1<<tdepth, file);
			fwrite(&thresholds[i], sizeof(float), 1, file);
		}

		fclose(file);
		printf("OK\n");
		fflush(stdout);
		return true;
	}

	float tsr;  // row scale ratio
	float tsc;  // column scale ratio
	int tdepth;  // max tree depth
	int ntrees;  // amount of trees

	int32_t tcodes[4096][1024];
	float luts[4096][1024];
	float thresholds[4096];
};
static Cascade cascade;

// hyperparameters
#define NRANDS 1024  // amount of binary test codes

// thread-safe random number generators
#define NUMPRNGS 1024
static uint64_t prngs[NUMPRNGS];

/*
	auxiliary stuff
*/

#define MAX(a, b) ((a)>(b)?(a):(b))
#define MIN(a, b) ((a)<(b)?(a):(b))
#define SQR(x) ((x)*(x))

/*
	portable time function
*/

#ifdef __GNUC__
#include <time.h>
float getticks()
{
	struct timespec ts;

	if(clock_gettime(CLOCK_MONOTONIC, &ts) < 0)
		return -1.0f;

	return ts.tv_sec + 1e-9f*ts.tv_nsec;
}
#else
#include <windows.h>
float getticks()
{
	static double freq = -1.0;
	LARGE_INTEGER lint;

	if(freq < 0.0)
	{
		if(!QueryPerformanceFrequency(&lint))
			return -1.0f;

		freq = lint.QuadPart;
	}

	if(!QueryPerformanceCounter(&lint))
		return -1.0f;

	return (float)( lint.QuadPart/freq );
}
#endif

/*
	multiply with carry PRNG
*/

uint32_t mwcrand_r(uint64_t* state)
{
	uint32_t* m = (uint32_t*)state;

	// bad state?
	if (m[0] == 0)
		m[0] = 0xAAAA;

	if (m[1] == 0)
		m[1] = 0xBBBB;

	// mutate state
	m[0] = 36969 * (m[0] & 65535) + (m[0] >> 16);
	m[1] = 18000 * (m[1] & 65535) + (m[1] >> 16);

	// output
	return (m[0] << 16) + m[1];
}

uint64_t prngglobal = 0x12345678000fffffLL;

void smwcrand(uint32_t seed)
{
	prngglobal = 0x12345678000fffffLL * seed;
}

uint32_t mwcrand()
{
	return mwcrand_r(&prngglobal);
}

void dump_floats(const std::string &filename, float *arr, int size)
{
	FILE *f = fopen(filename.c_str(), "wb");
	if (!f)
		return;

	for (int i = 0; i < size; ++i)
		fprintf(f, "%.5f\n", arr[i]);
}

/*
- loads an 8-bit grey image saved in the <RID> file format
- <RID> file contents:
	- a 32-bit signed integer h (image height)
	- a 32-bit signed integer w (image width)
	- an array of w*h unsigned bytes representing pixel intensities
	- image type (0: background, -1: hard negative, >0: positives)
	- if image type > 0, n times:
		- x
		- y
		- w
		- h
*/

bool load_image(std::vector<uint8_t> &pixels, int* nrows, int* ncols, FILE* file)
{
	if (fread(nrows, sizeof(int), 1, file) != 1)
		return false;

	if (fread(ncols, sizeof(int), 1, file) != 1)
		return false;

	pixels.resize(*nrows * *ncols);

	// read pixels
	if (fread(&pixels[0], sizeof(uint8_t), *nrows * *ncols, file) != *nrows * *ncols)
		return false;

	// we're done
	return true;
}

int load_training_data(const char* path)
{
	FILE* file = fopen(path, "rb");
	if (!file)
		return 0;

	int total_images = 0;
	while (load_image(
			dataset.ppixels[total_images],
			&dataset.pdims[total_images][0],
			&dataset.pdims[total_images][1],
			file))
	{
		int n = 0;
		if (fread(&n, sizeof(int), 1, file) != 1)
			return 1;

		if (n == 0)
			dataset.background.push_back(total_images);
		else if (n == -1)
			dataset.negatives.push_back(total_images);
		else
		{
			for (int i = 0; i < n; ++i)
			{
				Detection obj;
				fread(&obj.x, sizeof(int), 1, file); // x
				fread(&obj.y, sizeof(int), 1, file); // y
				fread(&obj.w, sizeof(int), 1, file); // w
				fread(&obj.h, sizeof(int), 1, file); // h

				obj.image_idx = total_images; // i
				dataset.objects.push_back(obj);
			}
		}

		++total_images;
	}

	return 1;
}

/*
	regression trees
*/

int bintest(int32_t tcode, int r, int c, int sr, int sc, int iind)
{
	int8_t* p = (int8_t*)&tcode;

	int r1 = (256*r + p[0]*sr)/256;
	int c1 = (256*c + p[1]*sc)/256;

	int r2 = (256*r + p[2]*sr)/256;
	int c2 = (256*c + p[3]*sc)/256;

	r1 = MIN(MAX(0, r1), dataset.pdims[iind][0]-1);
	c1 = MIN(MAX(0, c1), dataset.pdims[iind][1]-1);

	r2 = MIN(MAX(0, r2), dataset.pdims[iind][0]-1);
	c2 = MIN(MAX(0, c2), dataset.pdims[iind][1]-1);

	return dataset.ppixels[iind][r1 * dataset.pdims[iind][1]+c1] <=
			dataset.ppixels[iind][r2 * dataset.pdims[iind][1]+c2];
}

float get_split_error(int32_t tcode, const Detection *stage_objects,
	int srs[], int scs[], double ws[], int inds[], int indsnum)
{
	double wsum, wsum0, wsum1;
	double wtvalsum0, wtvalsumsqr0, wtvalsum1, wtvalsumsqr1;

	wsum = wsum0 = wsum1 = wtvalsum0 = wtvalsum1 = wtvalsumsqr0 = wtvalsumsqr1 = 0.0;

	for (int i = 0; i < indsnum; ++i)
	{
		if (bintest(tcode, stage_objects[inds[i]].y, stage_objects[inds[i]].x,
			srs[inds[i]], scs[inds[i]], stage_objects[inds[i]].image_idx))
		{
			wsum1 += ws[inds[i]];
			wtvalsum1 += ws[inds[i]] * stage_objects[inds[i]].obj_class;
			wtvalsumsqr1 += ws[inds[i]] * SQR(stage_objects[inds[i]].obj_class);
		}
		else
		{
			wsum0 += ws[inds[i]];
			wtvalsum0 += ws[inds[i]] * stage_objects[inds[i]].obj_class;
			wtvalsumsqr0 += ws[inds[i]] * SQR(stage_objects[inds[i]].obj_class);
		}

		wsum += ws[inds[i]];
	}

	double wmse0 = wtvalsumsqr0 - SQR(wtvalsum0) / wsum0;
	double wmse1 = wtvalsumsqr1 - SQR(wtvalsum1) / wsum1;

	return (float)((wmse0 + wmse1) / wsum);
}

int split_training_data(int32_t tcode, const Detection *stage_objects,
		int srs[], int scs[], double ws[], int inds[], int ninds)
{
	int stop = 0;
	int i = 0;
	int j = ninds - 1;

	while (!stop)
	{
		while (!bintest(tcode, stage_objects[inds[i]].y, stage_objects[inds[i]].x,
				srs[inds[i]], scs[inds[i]], stage_objects[inds[i]].image_idx))
		{
			if (i == j)
				break;
			else
				++i;
		}

		while (bintest(tcode, stage_objects[inds[j]].y, stage_objects[inds[j]].x,
				srs[inds[j]], scs[inds[j]], stage_objects[inds[j]].image_idx))
		{
			if (i == j)
				break;
			else
				--j;
		}

		if (i == j)
			stop = 1;
		else
			std::swap(inds[i], inds[j]);
	}

	int n0 = 0;
	for (i = 0; i < ninds; ++i)
		if (!bintest(tcode, stage_objects[inds[i]].y, stage_objects[inds[i]].x,
					srs[inds[i]], scs[inds[i]], stage_objects[inds[i]].image_idx))
			++n0;

	return n0;
}

int grow_subtree(int32_t tcodes[], float lut[], int nodeidx, int d, int maxd,
	const Detection *stage_objects, int srs[], int scs[],
	double ws[], int inds[], int ninds)
{
	if (d == maxd)
	{
		int lutidx = nodeidx - ((1<<maxd)-1);

		// compute output: a simple average
		double tvalaccum = 0.0;
		double wsum = 0.0;

		for(int i = 0; i < ninds; ++i)
		{
			tvalaccum += ws[inds[i]] * stage_objects[inds[i]].obj_class;
			wsum += ws[inds[i]];
		}

		if (wsum == 0.0)
			lut[lutidx] = 0.0f;
		else
			lut[lutidx] = (float)(tvalaccum / wsum);

		return 1;
	}
	else if (ninds <= 1)
	{
		tcodes[nodeidx] = 0;

		grow_subtree(tcodes, lut, 2*nodeidx+1, d+1, maxd, stage_objects, srs, scs, ws, inds, ninds);
		grow_subtree(tcodes, lut, 2*nodeidx+2, d+1, maxd, stage_objects, srs, scs, ws, inds, ninds);

		return 1;
	}

	// generate binary test codes
	int nrands = NRANDS;
	int32_t tmptcodes[2048];
	for (int i = 0; i < nrands; ++i)
		tmptcodes[i] = mwcrand();

	float es[2048];
	#pragma omp parallel for
	for (int i = 0; i < nrands; ++i)
		es[i] = get_split_error(tmptcodes[i], stage_objects, srs, scs, ws, inds, ninds);

	float e = es[0];
	tcodes[nodeidx] = tmptcodes[0];

	for (int i = 1; i < nrands; ++i)
		if(e > es[i])
		{
			e = es[i];
			tcodes[nodeidx] = tmptcodes[i];
		}

	int n0 = split_training_data(tcodes[nodeidx], stage_objects, srs, scs, ws, inds, ninds);

	grow_subtree(tcodes, lut, 2*nodeidx+1, d+1, maxd, stage_objects, srs, scs, ws, &inds[0], n0);
	grow_subtree(tcodes, lut, 2*nodeidx+2, d+1, maxd, stage_objects, srs, scs, ws, &inds[n0], ninds-n0);

	return 1;
}

int grow_rtree(int32_t tcodes[], float lut[], int d,
	const Detection *stage_objects,
	int srs[], int scs[], double ws[], int n)
{
	printf("	**growing tree... ");
	int* inds = (int*)malloc(n*sizeof(int));

	for (int i = 0; i < n; ++i)
		inds[i] = i;

	int ret = grow_subtree(tcodes, lut, 0, 0, d, stage_objects, srs, scs, ws, inds, n);
	free(inds);
	printf("OK\r");
	return ret;
}

float get_tree_output(int i, int r, int c, int sr, int sc, int iind)
{
	int idx = 1;

	for (int j = 0; j < cascade.tdepth; ++j)
		idx = 2*idx + bintest(cascade.tcodes[i][idx-1], r, c, sr, sc, iind);

	return cascade.luts[i][idx - (1 << cascade.tdepth)];
}

int classify_region(float* o, int r, int c, int w, int h, int iind)
{
	if (!cascade.ntrees)
		return 1;

	int sr = (int)(cascade.tsr * h);
	int sc = (int)(cascade.tsc * w);

	*o = 0.0f;
	for (int i = 0; i < cascade.ntrees; ++i)
	{
		*o += get_tree_output(i, r, c, sr, sc, iind);
		if (*o <= cascade.thresholds[i])
			return -1;
	}

	return 1;
}

int learn_new_stage(float mintpr, float maxfpr, int maxntrees,
	Detection *stage_objects, int np, int nn)
{
	printf("* learning stage %d...\n", ++cur_stage);
	fflush(stdout);

	int* srs = (int*)malloc((np+nn)*sizeof(int));
	int* scs = (int*)malloc((np+nn)*sizeof(int));

	for (int i = 0; i < np + nn; ++i)
	{
		srs[i] = int(cascade.tsr * stage_objects[i].h);
		scs[i] = int(cascade.tsc * stage_objects[i].w);
	}

	double* ws = (double*)malloc((np+nn)*sizeof(double));

	maxntrees += cascade.ntrees;
	float fpr = 1.0f;
	while (cascade.ntrees < maxntrees && fpr > maxfpr)
	{
		float t = getticks();

		// compute weights
		double wsum = 0.0;
		for (int i = 0; i < np + nn; ++i)
		{
			if (stage_objects[i].obj_class > 0)
				ws[i] = exp(-1.0 * stage_objects[i].score) / np;
			else
				ws[i] = exp(+1.0 * stage_objects[i].score) / nn;

			wsum += ws[i];
		}

		// normalize weights
		for (int i = 0; i < np + nn; ++i)
			ws[i] /= wsum;

		// grow a tree
		grow_rtree(
				cascade.tcodes[cascade.ntrees], cascade.luts[cascade.ntrees],
				cascade.tdepth, stage_objects,
				srs, scs, ws, np + nn);
		cascade.thresholds[cascade.ntrees++] = -1337.0f;

		// update outputs
		for (int i = 0; i < np + nn; ++i)
			stage_objects[i].score += get_tree_output(
						cascade.ntrees - 1,
						stage_objects[i].y,
						stage_objects[i].x,
						srs[i], scs[i], stage_objects[i].image_idx);

		// get threshold
		float threshold = 5.0f;
		float tpr = 0;
		int maxiter = 1000000;
		int it = 0;
		for (; it < maxiter && tpr < mintpr; ++it)
		{
			threshold -= 0.005f;

			int numtps = 0;
			int numfps = 0;

			for (int s = 0; s < np + nn; ++s)
			{
				if (stage_objects[s].obj_class > 0 && stage_objects[s].score > threshold)
					++numtps;
				if (stage_objects[s].obj_class < 0 && stage_objects[s].score > threshold)
					++numfps;
			}

			tpr = numtps / (float)np;
			fpr = numfps / (float)nn;
		}
		if (it == maxiter)
		{
			printf("	** MAX ITERATIONS\n");
			//dump_floats("os.dump", os, np + nn);
		}

		cascade.thresholds[cascade.ntrees - 1] = threshold;
		printf("	** tree %d (%d stage, %.2f s): th=%.2f, tpr=%f, fpr=%f\n",
				cascade.ntrees, cur_stage, getticks() - t, threshold, tpr, fpr);
		fflush(stdout);
	}

	printf("	** threshold set to %f\n", cascade.thresholds[cascade.ntrees - 1]);
	fflush(stdout);

	free(srs);
	free(scs);
	free(ws);

	return 1;
}

void collect_false_positives_random(
		const Dataset &dataset, Detection *stage_objects,
		int np,
		int &nn, int64_t &neg_tries, int64_t &total)
{
	int64_t neg_tries_limit = int64_t(np) * 100000;  // 1e-5 fpr for each thread
	bool have_enough_false_det = false;
	#pragma omp parallel
	{
		int thid = omp_get_thread_num();
		int neg_tries_thread = 0;  //  thread local nw counter

		// data mine hard negatives
		while (!have_enough_false_det)
		{
			// take random image
			int iind = dataset.background[
					mwcrand_r(&prngs[thid]) % dataset.background.size()];

			// sample the size of a random object in the pool
			int obj_num = mwcrand_r(&prngs[thid]) % dataset.objects.size();
			int obj_w = dataset.objects[obj_num].w;
			int obj_h = dataset.objects[obj_num].h;
			int obj_x = mwcrand_r(&prngs[thid]) % (dataset.pdims[iind][1] - obj_w);
			int obj_y = mwcrand_r(&prngs[thid]) % (dataset.pdims[iind][0] - obj_h);

			++neg_tries_thread;
			float o;
			if (classify_region(&o, obj_y, obj_x, obj_w, obj_h, iind) == 1)
			{
				// we have a false positive ...
				#pragma omp critical
				{
					if (nn < np && neg_tries_thread < neg_tries_limit)
					{
						stage_objects->x = obj_x;
						stage_objects->y = obj_y;
						stage_objects->w = obj_w;
						stage_objects->h = obj_h;
						stage_objects->image_idx = iind;
						stage_objects->obj_class = -1;
						stage_objects->score = 0;
						++stage_objects;

						++total;
						++nn;
					}
					else
						have_enough_false_det = true;
				}
			}

			#pragma omp master
			{
				if (neg_tries_thread % 100000 == 0)
				{
					printf("%.2lf %ld\r", o, neg_tries);
					fflush(stdout);
				}
			}
		}  // until have enough negatives or reach tries limit

		#pragma omp atomic
		neg_tries += neg_tries_thread;
	}  // omp parallel
}

//void collect_false_positives_detect(
//		const Dataset &dataset, Detection *stage_objects,
//		int &nn)
//{

//}

void collect_negatives_random(
		const Dataset &dataset, Detection *stage_objects,
		int np, int nn,
		int &cur_samples, int64_t &total_samples)
{
	for (int s = nn; s < np; ++s)
	{
		// take random image
		int iind = dataset.background[
				mwcrand_r(&prngs[0]) % dataset.background.size()];

		// sample the size of a random object in the pool
		int obj_num = mwcrand_r(&prngs[0]) % dataset.objects.size();
		int obj_w = dataset.objects[obj_num].w;
		int obj_h = dataset.objects[obj_num].h;
		int obj_x = mwcrand_r(&prngs[0]) % (dataset.pdims[iind][1] - obj_w);
		int obj_y = mwcrand_r(&prngs[0]) % (dataset.pdims[iind][0] - obj_h);
		stage_objects->x = obj_x;
		stage_objects->y = obj_y;
		stage_objects->w = obj_w;
		stage_objects->h = obj_h;
		stage_objects->image_idx = iind;
		stage_objects->obj_class = -1;
		stage_objects->score = 0;
		++stage_objects;
		++cur_samples;
		++total_samples;
	}
}

float sample_training_data(
		const Dataset &dataset, Detection *stage_objects, int* np, int* nn)
{
	printf("* sampling data...\n");
	fflush(stdout);

	int t = getticks();
	int64_t total = 0;

	// TODO: add ALL positives to dataset (or/and export doubtful samples for review)
	// object samples
	printf("* sampling positives...\n");
	fflush(stdout);
	for (const auto &obj: dataset.objects)
	{
		if (classify_region(&stage_objects->score,
					obj.y, obj.x, obj.w, obj.h,
					obj.image_idx) == 1)
		{
			stage_objects->x = obj.x;
			stage_objects->y = obj.y;
			stage_objects->w = obj.w;
			stage_objects->h = obj.h;
			stage_objects->image_idx = obj.image_idx;
			stage_objects->obj_class = 1;
			++stage_objects;
			++total;
		}
	}

	*np = int(total);
	printf("Got %d positive samples\n", *np);

	// non-object samples
	*nn = 0;

	// sample hard negatives (i. e. manually given false positives)
	printf("* sampling hard negatives...\n");
	int hard_negatives = 0;
	for (const auto &obj: dataset.negatives)
	{
		// whole image
		stage_objects->x = 0;
		stage_objects->y = 0;
		stage_objects->w = dataset.pdims[obj][1] - 1;
		stage_objects->h = dataset.pdims[obj][0] - 1;
		stage_objects->image_idx = obj;
		stage_objects->obj_class = -1;
		stage_objects->score = 0;
		++stage_objects;

		++total;
		++hard_negatives;
	}
	printf("Got %d hard negative samples\n", *nn);

	// TODO: detected negatives from non-object images

	// random export from non-object images
	printf("* sampling negatives...\n");
	fflush(stdout);
	int random_negatives = 0;
	int64_t neg_tries = 0;
	if (!dataset.background.empty())
	{
		collect_false_positives_random(
				dataset, stage_objects, *np, *nn, neg_tries, total);
		// get random samples if we have not ehough negatives
		collect_negatives_random(
				dataset, stage_objects, *np, *nn, random_negatives, total);
	}
	else
		neg_tries = 1;  // just division by zero prevention

	float etpr = *np / (float)dataset.objects.size();
	float efpr = (float)(*nn / (double)neg_tries);

	printf("* sampling finished (totally %lld samples)\n", (long long int)total);
	printf("	** elapsed time: %.2f s\n", getticks() - t);
	printf("	** cascade TPR=%.8f (%d/%d)\n", etpr, *np, int(dataset.objects.size()));
	printf("	** cascade FPR=%.8f (%d/%lld)\n", efpr, *nn, (long long int)neg_tries);
	fflush(stdout);

	*nn += hard_negatives;
	*nn += random_negatives;

	return efpr;
}

static Detection stage_objects[2 * MAX_N];

bool learn_with_default_parameters(const char* trdata, const char* dst, float tfpr)
{
	if (!load_training_data(trdata))
	{
		printf("* cannot load training data\n");
		return false;
	}

	if (cascade.load_from_file(dst))
		printf("Cascade '%s' loaded, continuing training\n", dst);
	else if (!cascade.save_to_file(dst))
	{
		printf("* cannot save cascade to '%s', ABORTING\n", dst);
		return false;
	}

	int np, nn;
	sample_training_data(dataset, stage_objects, &np, &nn);
	learn_new_stage(0.9800f, 0.5f, 4, stage_objects, np, nn);
	cascade.save_to_file(dst);
	printf("\n");

	sample_training_data(dataset, stage_objects, &np, &nn);
	learn_new_stage(0.9850f, 0.5f, 8, stage_objects, np, nn);
	cascade.save_to_file(dst);
	printf("\n");

	sample_training_data(dataset, stage_objects, &np, &nn);
	learn_new_stage(0.9900f, 0.5f, 16, stage_objects, np, nn);
	cascade.save_to_file(dst);
	printf("\n");

	sample_training_data(dataset, stage_objects, &np, &nn);
	learn_new_stage(0.9950f, 0.5f, 32, stage_objects, np, nn);
	cascade.save_to_file(dst);
	printf("\n");

	while (sample_training_data(dataset, stage_objects, &np, &nn) > tfpr)
	{
		learn_new_stage(0.9975f, 0.5f, 64, stage_objects, np, nn);
		cascade.save_to_file(dst);
		printf("\n");
	}

	printf("* target FPR achieved, terminating the learning process\n");
	return true;
}

void usage(const char *prog_name)
{
	printf("Usage:\n");
	printf("%s [-sr scale_rows] [-sc scale_col] [--depth max_tree_depth] "
		   "[--init-only] [--one-stage] "
		   "[--tpr required_TPR] [--fpr required_FPR] [--ntrees] "
		   "<data file> <output file>\n", prog_name);
}

int main(int argc, char* argv[])
{
	// initialize the PRNG
	smwcrand(time(0));
	// initialize a PRNG for each thread
	for (int i = 0; i < NUMPRNGS; ++i)
		prngs[i] = 0xFFFF * mwcrand() + 0xFFFF1234FFFF0001LL * mwcrand();

	std::string data_file_name;
	std::string cascade_file_name;
	bool init_only = false;
	bool one_stage = false;
	float tpr = 0;
	float fpr = 0;
	int ntrees = 0;
	int opt_count = 1;
	while (opt_count < argc)
	{
		if (std::string(argv[opt_count]) == "-h")
		{
			usage(argv[0]);
			return 0;
		}
		else if (std::string(argv[opt_count]) == "--sr")
		{
			++opt_count;
			if (opt_count < argc)
				cascade.tsr = float(atof(argv[opt_count + 1]));
		}
		else if (std::string(argv[opt_count]) == "--sc")
		{
			++opt_count;
			if (opt_count < argc)
				cascade.tsc = float(atof(argv[opt_count + 1]));
		}
		else if (std::string(argv[opt_count]) == "--depth")
		{
			++opt_count;
			if (opt_count < argc)
				cascade.tdepth = atoi(argv[opt_count + 1]);
		}
		else if (std::string(argv[opt_count]) == "--tpr")
		{
			++opt_count;
			if (opt_count < argc)
				tpr = float(atof(argv[opt_count + 1]));
		}
		else if (std::string(argv[opt_count]) == "--fpr")
		{
			++opt_count;
			if (opt_count < argc)
				fpr = float(atof(argv[opt_count + 1]));
		}
		else if (std::string(argv[opt_count]) == "--ntrees")
		{
			++opt_count;
			if (opt_count < argc)
				cascade.ntrees = atoi(argv[opt_count + 1]);
		}
		else if (std::string(argv[opt_count]) == "--init-only")
		{
			init_only = true;
		}
		else if (std::string(argv[opt_count]) == "--one-stage")
		{
			one_stage = true;
		}
		else if (argv[opt_count][0] == '-')
		{
			printf("unknown parameter %s\n", argv[opt_count]);
		}
		else if (data_file_name.empty())
		{
			data_file_name = argv[opt_count];
		}
		else if (cascade_file_name.empty())
		{
			cascade_file_name = argv[opt_count];
		}
		else
		{
			printf("unknown parameter %s\n", argv[opt_count]);
		}
		++opt_count;
	}

	if (data_file_name.empty() || cascade_file_name.empty())
	{
		usage(argv[0]);
		return -1;
	}

	if (init_only)
	{
		ntrees = 0;
		if (!cascade.save_to_file(cascade_file_name.c_str()))
			return -1;

		printf("* initializing: (%f, %f, %d)\n",
				cascade.tsr, cascade.tsc, cascade.tdepth);
		return 0;
	}
	else if (one_stage)
	{
		if (!cascade.load_from_file(cascade_file_name.c_str()))
		{
			printf("* cannot load a cascade from '%s', creating new one\n",
				   cascade_file_name.c_str());
			if (!cascade.save_to_file(cascade_file_name.c_str()))
			{
				printf("* cannot save cascade to '%s', ABORTING\n",
						cascade_file_name.c_str());
				return -1;
			}
		}

		if (!load_training_data(data_file_name.c_str()))
		{
			printf("* cannot load the training data from '%s'\n",
				   data_file_name.c_str());
			return 1;
		}

		int np, nn;
		sample_training_data(dataset, stage_objects, &np, &nn);
		learn_new_stage(tpr, fpr, ntrees, stage_objects, np, nn);

		if (!cascade.save_to_file(cascade_file_name.c_str()))
		{
			printf("* cannot save cascade to '%s', ABORTING\n",
					cascade_file_name.c_str());
			return -1;
		}
	}
	else
		learn_with_default_parameters(data_file_name.c_str(), cascade_file_name.c_str(), 1e-6f);

	return 0;
}
