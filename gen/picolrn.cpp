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

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <malloc.h>
#include <stdint.h>

// hyperparameters
#define NRANDS 1024

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
	uint32_t* m;

	//
	m = (uint32_t*)state;

	// bad state?
	if(m[0] == 0)
		m[0] = 0xAAAA;

	if(m[1] == 0)
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
	prngglobal = 0x12345678000fffffLL*seed;
}

uint32_t mwcrand()
{
	return mwcrand_r(&prngglobal);
}

/*
	
*/

#define MAX_N 2000000

int N = 0;
uint8_t* ppixels[MAX_N];
int pdims[MAX_N][2]; // (nrows, ncols)

int nbackground = 0;
int background[MAX_N]; // i

int nobjects = 0;
int objects[MAX_N][4]; // (r, c, s, i)

int load_image(uint8_t* pixels[], int* nrows, int* ncols, FILE* file)
{
	/*
	- loads an 8-bit grey image saved in the <RID> file format
	- <RID> file contents:
		- a 32-bit signed integer h (image height)
		- a 32-bit signed integer w (image width)
		- an array of w*h unsigned bytes representing pixel intensities
	*/

	if (fread(nrows, sizeof(int), 1, file) != 1)
		return 0;

	if (fread(ncols, sizeof(int), 1, file) != 1)
		return 0;

	*pixels = (uint8_t*)malloc(*nrows**ncols*sizeof(uint8_t));
	if (!*pixels)
		return 0;

	// read pixels
	if (fread(*pixels, sizeof(uint8_t), *nrows**ncols, file) != *nrows**ncols)
		return 0;

	// we're done
	return 1;
}

int load_training_data(const char* path)
{
	FILE* file = fopen(path, "rb");
	if (!file)
		return 0;

	N = 0;

	nbackground = 0;
	nobjects = 0;

	while (load_image(&ppixels[N], &pdims[N][0], &pdims[N][1], file))
	{
		int n = 0;
		if (fread(&n, sizeof(int), 1, file) != 1)
			return 1;

		if (!n)
		{
			background[nbackground] = N;
			++nbackground;
		}
		else
		{
			for(int i = 0; i < n; ++i)
			{
				fread(&objects[nobjects][0], sizeof(int), 1, file); // r
				fread(&objects[nobjects][1], sizeof(int), 1, file); // c
				fread(&objects[nobjects][2], sizeof(int), 1, file); // s

				objects[nobjects][3] = N; // i
				++nobjects;
			}
		}

		++N;
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

	r1 = MIN(MAX(0, r1), pdims[iind][0]-1);
	c1 = MIN(MAX(0, c1), pdims[iind][1]-1);

	r2 = MIN(MAX(0, r2), pdims[iind][0]-1);
	c2 = MIN(MAX(0, c2), pdims[iind][1]-1);

	return ppixels[iind][r1*pdims[iind][1]+c1]<=ppixels[iind][r2*pdims[iind][1]+c2];
}

float get_split_error(int32_t tcode, float tvals[], int rs[], int cs[], int srs[], int scs[], int iinds[], double ws[], int inds[], int indsnum)
{
	int i, j;

	double wsum, wsum0, wsum1;
	double wtvalsum0, wtvalsumsqr0, wtvalsum1, wtvalsumsqr1;

	double wmse0, wmse1;

	//
	wsum = wsum0 = wsum1 = wtvalsum0 = wtvalsum1 = wtvalsumsqr0 = wtvalsumsqr1 = 0.0;

	for(i=0; i<indsnum; ++i)
	{
		if( bintest(tcode, rs[inds[i]], cs[inds[i]], srs[inds[i]], scs[inds[i]], iinds[inds[i]]) )
		{
			wsum1 += ws[inds[i]];
			wtvalsum1 += ws[inds[i]]*tvals[inds[i]];
			wtvalsumsqr1 += ws[inds[i]]*SQR(tvals[inds[i]]);
		}
		else
		{
			wsum0 += ws[inds[i]];
			wtvalsum0 += ws[inds[i]]*tvals[inds[i]];
			wtvalsumsqr0 += ws[inds[i]]*SQR(tvals[inds[i]]);
		}

		wsum += ws[inds[i]];
	}

	//
	wmse0 = wtvalsumsqr0 - SQR(wtvalsum0)/wsum0;
	wmse1 = wtvalsumsqr1 - SQR(wtvalsum1)/wsum1;

	//
	return (float)( (wmse0 + wmse1)/wsum );
}

int split_training_data(int32_t tcode, float tvals[], int rs[], int cs[], int srs[], int scs[], int iinds[], double ws[], int inds[], int ninds)
{
	int stop;
	int i, j;

	int n0;

	stop = 0;

	i = 0;
	j = ninds - 1;

	while(!stop)
	{
		//
		while( !bintest(tcode, rs[inds[i]], cs[inds[i]], srs[inds[i]], scs[inds[i]], iinds[inds[i]]) )
		{
			if( i==j )
				break;
			else
				++i;
		}

		while( bintest(tcode, rs[inds[j]], cs[inds[j]], srs[inds[j]], scs[inds[j]], iinds[inds[j]]) )
		{
			if( i==j )
				break;
			else
				--j;
		}

		if( i==j )
			stop = 1;
		else
		{
			// swap
			inds[i] = inds[i] ^ inds[j];
			inds[j] = inds[i] ^ inds[j];
			inds[i] = inds[i] ^ inds[j];
		}
	}

	n0 = 0;

	for(i=0; i<ninds; ++i)
		if( !bintest(tcode, rs[inds[i]], cs[inds[i]], srs[inds[i]], scs[inds[i]], iinds[inds[i]]) )
			++n0;

	return n0;
}

int grow_subtree(int32_t tcodes[], float lut[], int nodeidx, int d, int maxd, float tvals[], int rs[], int cs[], int srs[], int scs[], int iinds[], double ws[], int inds[], int ninds)
{
	int i, nrands;

	int32_t tmptcodes[2048];
	float es[2048], e;

	int n0;

	if (d == maxd)
	{
		int lutidx;
		double tvalaccum, wsum;

		//
		lutidx = nodeidx - ((1<<maxd)-1);

		// compute output: a simple average
		tvalaccum = 0.0;
		wsum = 0.0;

		for(i=0; i<ninds; ++i)
		{
			tvalaccum += ws[inds[i]]*tvals[inds[i]];
			wsum += ws[inds[i]];
		}

		if(wsum == 0.0)
			lut[lutidx] = 0.0f;
		else
			lut[lutidx] = (float)( tvalaccum/wsum );

		return 1;
	}
	else if (ninds <= 1)
	{
		tcodes[nodeidx] = 0;

		grow_subtree(tcodes, lut, 2*nodeidx+1, d+1, maxd, tvals, rs, cs, srs, scs, iinds, ws, inds, ninds);
		grow_subtree(tcodes, lut, 2*nodeidx+2, d+1, maxd, tvals, rs, cs, srs, scs, iinds, ws, inds, ninds);

		return 1;
	}

	// generate binary test codes
	nrands = NRANDS;

	for(i=0; i<nrands; ++i)
		tmptcodes[i] = mwcrand();

	#pragma omp parallel for
	for (i=0; i<nrands; ++i)
		es[i] = get_split_error(tmptcodes[i], tvals, rs, cs, srs, scs, iinds, ws, inds, ninds);

	e = es[0];
	tcodes[nodeidx] = tmptcodes[0];

	for(i=1; i<nrands; ++i)
		if(e > es[i])
		{
			e = es[i];
			tcodes[nodeidx] = tmptcodes[i];
		}

	n0 = split_training_data(tcodes[nodeidx], tvals, rs, cs, srs, scs, iinds, ws, inds, ninds);

	grow_subtree(tcodes, lut, 2*nodeidx+1, d+1, maxd, tvals, rs, cs, srs, scs, iinds, ws, &inds[0], n0);
	grow_subtree(tcodes, lut, 2*nodeidx+2, d+1, maxd, tvals, rs, cs, srs, scs, iinds, ws, &inds[n0], ninds-n0);

	return 1;
}

int grow_rtree(int32_t tcodes[], float lut[], int d, float tvals[], int rs[], int cs[], int srs[], int scs[], int iinds[], double ws[], int n)
{
	printf("	**growing tree... ");
	int* inds = (int*)malloc(n*sizeof(int));

	for (int i = 0; i < n; ++i)
		inds[i] = i;

	int ret = grow_subtree(tcodes, lut, 0, 0, d, tvals, rs, cs, srs, scs, iinds, ws, inds, n);
	free(inds);
	printf("OK\n");
	return ret;
}

float tsr, tsc;
int tdepth;
int ntrees=0;

int32_t tcodes[4096][1024];
float luts[4096][1024];

float thresholds[4096];

int load_cascade_from_file(const char* path)
{
	FILE* file = fopen(path, "rb");
	if (!file)
		return 0;

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
	return 1;
}

int save_cascade_to_file(const char* path)
{
	printf("* saving cascade...");
	fflush(stdout);
	FILE* file = fopen(path, "wb");
	if (!file)
		return 0;

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
	return 1;
}

float get_tree_output(int i, int r, int c, int sr, int sc, int iind)
{
	int idx = 1;

	for (int j=0; j < tdepth; ++j)
		idx = 2*idx + bintest(tcodes[i][idx-1], r, c, sr, sc, iind);

	return luts[i][idx - (1<<tdepth)];
}

int classify_region(float* o, int r, int c, int s, int iind)
{
	if (!ntrees)
		return 1;

	int sr = (int)(tsr*s);
	int sc = (int)(tsc*s);

	*o = 0.0f;

	for (int i = 0; i < ntrees; ++i)
	{
		*o += get_tree_output(i, r, c, sr, sc, iind);
		if (*o <= thresholds[i])
			return -1;
	}

	return 1;
}

int learn_new_stage(float mintpr, float maxfpr, int maxntrees, float tvals[],
					int rs[], int cs[], int ss[], int iinds[], float os[], int np, int nn)
{
	printf("* learning new stage...\n");
	fflush(stdout);

	int* srs = (int*)malloc((np+nn)*sizeof(int));
	int* scs = (int*)malloc((np+nn)*sizeof(int));

	for (int i = 0; i < np + nn; ++i)
	{
		srs[i] = (int)( tsr*ss[i] );
		scs[i] = (int)( tsc*ss[i] );
	}

	double* ws = (double*)malloc((np+nn)*sizeof(double));

	maxntrees += ntrees;
	float fpr = 1.0f;
	float threshold = 5.0f;
	while (ntrees < maxntrees && fpr > maxfpr)
	{
		float t = getticks();

		// compute weights ...
		double wsum = 0.0;
		for (int i=0; i<np+nn; ++i)
		{
			if(tvals[i] > 0)
				ws[i] = exp(-1.0*os[i])/np;
			else
				ws[i] = exp(+1.0*os[i])/nn;

			wsum += ws[i];
		}

		for (int i = 0; i < np + nn; ++i)
			ws[i] /= wsum;

		// grow a tree ...
		grow_rtree(tcodes[ntrees], luts[ntrees], tdepth, tvals, rs, cs, srs, scs, iinds, ws, np+nn);
		thresholds[ntrees] = -1337.0f;
		++ntrees;

		// update outputs ...
		for (int i = 0; i < np + nn; ++i)
		{
			float o = get_tree_output(ntrees-1, rs[i], cs[i], srs[i], scs[i], iinds[i]);
			os[i] += o;
		}

		// get threshold ...
		float threshold = 5.0f;
		float tpr = 0;
		do
		{
			threshold -= 0.005f;

			int numtps = 0;
			int numfps = 0;

			for (int i = 0; i < np + nn; ++i)
			{
				if (tvals[i] > 0 && os[i] > threshold)
					++numtps;
				if (tvals[i] < 0 && os[i] > threshold)
					++numfps;
			}

			tpr = numtps / (float)np;
			fpr = numfps / (float)nn;
		}
		while (tpr < mintpr);

		printf("	** tree %d (%d [s]) ... stage tpr=%f, stage fpr=%f\n",
			   ntrees, (int)(getticks()-t), tpr, fpr);
		fflush(stdout);
	}

	thresholds[ntrees-1] = threshold;
	printf("	** threshold set to %f\n", threshold);
	fflush(stdout);

	free(srs);
	free(scs);
	free(ws);

	return 1;
}

float sample_training_data(float tvals[], int rs[], int cs[], int ss[],
						   int iinds[], float os[], int* np, int* nn)
{
	printf("* sampling data...\n");
	fflush(stdout);

	#define NUMPRNGS 1024
	static int prngsinitialized = 0;
	static uint64_t prngs[NUMPRNGS];

	int t = getticks();
	int n = 0;

	// object samples
	printf("* sampling positives...\n");
	fflush(stdout);
	for (int i = 0; i < nobjects; ++i)
	{
		if (classify_region(&os[n], objects[i][0], objects[i][1], objects[i][2], objects[i][3]) == 1)
		{
			rs[n] = objects[i][0];
			cs[n] = objects[i][1];
			ss[n] = objects[i][2];

			iinds[n] = objects[i][3];
			tvals[n] = +1;

			++n;
		}
	}

	*np = n;

	// non-object samples
	if (!prngsinitialized)
	{
		// initialize a PRNG for each thread
		for (int i=0; i<NUMPRNGS; ++i)
			prngs[i] = 0xFFFF*mwcrand() + 0xFFFF1234FFFF0001LL*mwcrand();

		prngsinitialized = 1;
	}

	int64_t nw = 0;
	*nn = 0;

	printf("* sampling negatives\n");
	fflush(stdout);
	int stop = 0;
	if (nbackground)
	{
		#pragma omp parallel
		{
			int thid = omp_get_thread_num();

			// data mine hard negatives
			while (!stop)
			{
				// take random image
				int iind = background[mwcrand_r(&prngs[thid]) % nbackground];

				// sample the size of a random object in the pool
				int r = mwcrand_r(&prngs[thid])%pdims[iind][0];
				int c = mwcrand_r(&prngs[thid])%pdims[iind][1];
				int s = objects[mwcrand_r(&prngs[thid])%nobjects][2];

				float o;
				if (classify_region(&o, r, c, s, iind) == 1)
				{
					// we have a false positive ...
					#pragma omp critical
					{
						if (*nn < *np)
						{
							rs[n] = r;
							cs[n] = c;
							ss[n] = s;

							iinds[n] = iind;

							os[n] = o;

							tvals[n] = -1;

							++n;
							++*nn;
						}
						else
							stop = 1;
					}
				}

				#pragma omp master
				{
					if (nw % 1000 == 0)
					{
						printf(".");
						fflush(stdout);
					}
				}

				if (!stop)
				{
					#pragma omp atomic
					++nw;
				}
			}
		}  // omp parallel
	}
	else
		nw = 1;

	/*
		print the estimated true positive and false positive rates
	*/

	float etpr = *np/(float)nobjects;
	float efpr = (float)( *nn/(double)nw );

	printf("\n* sampling finished\n");
	printf("	** elapsed time: %d\n", (int)(getticks()-t));
	printf("	** cascade TPR=%.8f\n", etpr);
	printf("	** cascade FPR=%.8f (%d/%lld)\n", efpr, *nn, (long long int)nw);
	fflush(stdout);

	return efpr;
}


static int rs[2*MAX_N];
static int cs[2*MAX_N];
static int ss[2*MAX_N];
static int iinds[2*MAX_N];
static float tvals[2*MAX_N];
static float os[2*MAX_N];

bool learn_with_default_parameters(const char* trdata, const char* dst)
{
	if (!load_training_data(trdata))
	{
		printf("* cannot load training data ...\n");
		return false;
	}

	if (!save_cascade_to_file(dst))
		return false;

	int np, nn;
	sample_training_data(tvals, rs, cs, ss, iinds, os, &np, &nn);
	learn_new_stage(0.9800f, 0.5f, 4, tvals, rs, cs, ss, iinds, os, np, nn);
	save_cascade_to_file(dst);
	printf("\n");

	sample_training_data(tvals, rs, cs, ss, iinds, os, &np, &nn);
	learn_new_stage(0.9850f, 0.5f, 8, tvals, rs, cs, ss, iinds, os, np, nn);
	save_cascade_to_file(dst);
	printf("\n");

	sample_training_data(tvals, rs, cs, ss, iinds, os, &np, &nn);
	learn_new_stage(0.9900f, 0.5f, 16, tvals, rs, cs, ss, iinds, os, np, nn);
	save_cascade_to_file(dst);
	printf("\n");

	sample_training_data(tvals, rs, cs, ss, iinds, os, &np, &nn);
	learn_new_stage(0.9950f, 0.5f, 32, tvals, rs, cs, ss, iinds, os, np, nn);
	save_cascade_to_file(dst);
	printf("\n");

	while (sample_training_data(tvals, rs, cs, ss, iinds, os, &np, &nn) > 1e-6f)
	{
		learn_new_stage(0.9975f, 0.5f, 64, tvals, rs, cs, ss, iinds, os, np, nn);
		save_cascade_to_file(dst);
		printf("\n");
	}

	printf("* target FPR achieved ... terminating the learning process ...\n");
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

	std::string data_file_name;
	std::string cascade_file_name;
	bool init_only = false;
	bool one_stage = false;
	tsr = 1.0f;  // scale row
	tsc = 1.0f;  // scale col
	tdepth = 5;  // tree max depth
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
				tsr = float(atof(argv[opt_count + 1]));
		}
		else if (std::string(argv[opt_count]) == "--sc")
		{
			++opt_count;
			if (opt_count < argc)
				tsc = float(atof(argv[opt_count + 1]));
		}
		else if (std::string(argv[opt_count]) == "--depth")
		{
			++opt_count;
			if (opt_count < argc)
				tdepth = atoi(argv[opt_count + 1]);
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
				ntrees = atoi(argv[opt_count + 1]);
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
		if (!save_cascade_to_file(cascade_file_name.c_str()))
			return 0;

		printf("* initializing: (%f, %f, %d)\n", tsr, tsc, tdepth);
		return 0;
	}
	else if (one_stage)
	{
		if (!load_cascade_from_file(cascade_file_name.c_str()))
		{
			printf("* cannot load a cascade from '%s', creating new one\n",
				   cascade_file_name.c_str());
			save_cascade_to_file(cascade_file_name.c_str());
		}

		if (!load_training_data(data_file_name.c_str()))
		{
			printf("* cannot load the training data from '%s'\n",
				   data_file_name.c_str());
			return 1;
		}

		int np, nn;
		sample_training_data(tvals, rs, cs, ss, iinds, os, &np, &nn);
		learn_new_stage(tpr, fpr, ntrees, tvals, rs, cs, ss, iinds, os, np, nn);

		if (!save_cascade_to_file(cascade_file_name.c_str()))
			return 1;
	}
	else
		learn_with_default_parameters(data_file_name.c_str(), cascade_file_name.c_str());

	return 0;
}
