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

#define MAX(a, b) ((a)>(b)?(a):(b))
#define MIN(a, b) ((a)<(b)?(a):(b))

#include "picornt.h"

#include <cstring>

int find_objects(
	float *rs, float *cs, float *ss, float *qs, int maxndetections,
	int (*detection_func)(float*, int, int, int, const uint8_t*, int, int, int),
	const uint8_t* pixels, int nrows, int ncols, int ldim,
	float scalefactor, float stridefactor, float minsize, float maxsize)
{
	float s;
	int ndetections;

	ndetections = 0;
	s = minsize;

	while (s<=maxsize)
	{
		float dr, dc;
		dr = dc = MAX(stridefactor*s, 1.0f);

		for (float r=s/2+1; r<=nrows-s/2-1; r+=dr)
		{
			for (float c=s/2+1; c<=ncols-s/2-1; c+=dc)
			{
				float q;
				if (detection_func(&q, r, c, s, pixels, nrows, ncols, ldim) != 1
						|| ndetections >= maxndetections)
					continue;

				qs[ndetections] = q;
				rs[ndetections] = r;
				cs[ndetections] = c;
				ss[ndetections] = s;

				++ndetections;
			}
		}

		s = scalefactor*s;
	}

	return ndetections;
}

float get_overlap(float r1, float c1, float s1, float r2, float c2, float s2)
{
	float overr = MAX(0, MIN(r1+s1/2, r2+s2/2) - MAX(r1-s1/2, r2-s2/2));
	float overc = MAX(0, MIN(c1+s1/2, c2+s2/2) - MAX(c1-s1/2, c2-s2/2));

	return overr*overc/(s1*s1+s2*s2-overr*overc);
}

void ccdfs(int a[], int i, float rs[], float cs[], float ss[], int n)
{
	for (int j = 0; j < n; ++j)
	{
		if (a[j] == 0 && get_overlap(rs[i], cs[i], ss[i], rs[j], cs[j], ss[j])>0.3f)
		{
			a[j] = a[i];

			ccdfs(a, j, rs, cs, ss, n);
		}
	}
}

int find_connected_components(int a[], float rs[], float cs[], float ss[], int n)
{
	if (!n)
		return 0;

	memset(a, 0, n * sizeof(float));

	int ncc = 0;
	int cc = 1;
	for (int i = 0; i < n; ++i)
	{
		if (a[i] != 0)
			continue;
		a[i] = cc;
		ccdfs(a, i, rs, cs, ss, n);
		++ncc;
		++cc;
	}

	return ncc;
}

int cluster_detections(float *rs, float *cs, float *ss, float *qs, int n)
{
	int a[4096];
	int ncc = find_connected_components(a, rs, cs, ss, n);
	if (!ncc)
		return 0;

	int idx = 0;
	for (int cc = 1; cc <= ncc; ++cc)
	{
		float sumqs=0.0f, sumrs=0.0f, sumcs=0.0f, sumss=0.0f;

		int k = 0;
		for (int i = 0; i < n; ++i)
		{
			if (a[i] != cc)
				continue;

			sumqs += qs[i];
			sumrs += rs[i];
			sumcs += cs[i];
			sumss += ss[i];
			++k;
		}

		qs[idx] = sumqs;  // accumulated confidence measure

		rs[idx] = sumrs/k;
		cs[idx] = sumcs/k;
		ss[idx] = sumss/k;

		++idx;
	}

	return idx;
}
