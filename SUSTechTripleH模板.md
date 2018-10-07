# SUSTechTripleH模板
[TOC]
##图论
###tarjan求割点
```C++
void tarjan(int u, int f) {
​	dfn[u] = low[u] = ++idx; int ch = 0;
​	reg(i, u, v) if(v != f) {
​		if(!dfn[v]) {
			ch++; dfs(v, u); 
​			tarjan(v); low[u] = min(low[u], low[v]);
			if((f == 0 && ch >= 2) || (f && low[v] >= dfn[u])) {
				tot += vis[u] ^ 1; vis[u] = 1;
			}
​		} else {
​			low[u] = min(low[u], dfn[v]);
​		}
​	}
}
```

###割边
```C++
void tarjan(int u, int f) {
​	dfn[u] = low[u] = ++idx; int ch = 0;
​	reg(i, u, v) if(v != f) {
​		if(!dfn[v]) {
			ch++; dfs(v, u); 
​			tarjan(v); low[u] = min(low[u], low[v]);
​			if(low[v] > dfn[u]); // u->v为割边
​		} else {
​			low[u] = min(low[u], dfn[v]);
​		}
​	}
}
```

###tarjan求强联通分量
```C++
struct edge {int to, next;}e[maxm];
int dfn[maxn], low[maxn], st[maxn], head[maxn], idx = 0, cnt = 0, scnt = 0, top;
int u[maxm], v[maxm], x[maxn], y[maxn], r[maxn], c[maxn], id[maxn], cost[maxn];
bool vis[maxn], in[maxn];

void insert(int u, int v) {
​	e[++cnt] = (edge){v, head[u]}; head[u] = cnt;
}

void tarjan(int u) {
​	dfn[u] = low[u] = ++idx;
​	st[++top] = u; vis[u] = 1;
​	reg(i, u, v) {
​		if(!dfn[v]) {
​			tarjan(v); low[u] = min(low[u], low[v]);
​		} else if(vis[v]) {
​			low[u] = min(low[u], dfn[v]);
​		}
​	}
​	if(dfn[u] == low[u]) {
​		++scnt; int v;
​		do {
​			v = st[top--]; vis[v] = 0;
​			id[v] = scnt;
​		} while(u != v);
​	}
}

void init() {
​	cnt = scnt = idx = top = 0;
​	clr(low); clr(dfn); clr(vis); clr(in); clr(id); clr(head);
​	memset(cost, 0x3f, sizeof(cost));
}
```

###A*求K短路
```C++
namespace A_star {
	const int maxn = 1010, maxm = 100010;

	int cnt, head[maxn], head1[maxn], dis[maxn]; bool vis[maxn];
	int s, t, k, n, m;

	struct edge {
		int to, next, w;
		bool operator < (const edge & rhs) const {
			return w > rhs.w;
		}
	} e[maxm], eg[maxm];

	struct node {
		int to, a, b;
		bool operator< (const node & rhs) const {
			if(a == rhs.a) return b > rhs.b;
			return a > rhs.a;
		}
	};

	void init() {
		cnt = 0; clr(head); clr(head1); clr(vis);
	}

	void ins(int u, int v, int w) {
		e[++cnt] = (edge) {u, head[v], w}; head[v] = cnt;
		eg[cnt] = (edge) {v, head1[u], w}; head1[u] = cnt;
	}

	void dijkstra(int st) {
		priority_queue<edge> pq;
		memset(dis, 0x3f, sizeof(dis));
		clr(vis); edge S = (edge) {st, 0, 0};
		dis[st] = 0; vis[st] = 1; pq.push(S); 
		while(!pq.empty()) {
			edge u = pq.top(); pq.pop(); vis[u.to] = 0;
			for(int i = head[u.to]; i; i = e[i].next) {
				if(dis[e[i].to] > dis[u.to] + e[i].w) {
					dis[e[i].to] = dis[u.to] + e[i].w;
					if(!vis[e[i].to]) {
						pq.push(e[i]); vis[e[i].to] = 1;
					}
				}
			}
		}
		while(!pq.empty()) pq.pop();
	}

	int a_star(int st, int ed, int k) {
		int cnt = 0; if(s == t) k++;
		if(dis[st] > T) return T + 1;
		priority_queue<node> pq;
		node S = (node) {st, dis[st], 0}; pq.push(S);
		while(!pq.empty()) {
			node cur = pq.top(); pq.pop();
			if(cur.to == ed) cnt++;
			if(cnt == k) return cur.b;
			for(int i = head1[cur.to]; i; i = eg[i].next) {
				node nxt; nxt.to = eg[i].to;
				nxt.b = cur.b + eg[i].w;
				nxt.a = nxt.b + dis[nxt.to];
				pq.push(nxt);
			}
		}
		return T + 1;
	}

	void work() {
		dijkstra(t); int ans = a_star(s, t, k);
	}
}
```


###最小树形图

```C++
#include <cstdio>
#include <cstring>
#include <algorithm>

using namespace std;

const int maxn = 100010;

int g[maxn], id[maxn], n, m, q, rt; double c[maxn], low[maxn], vis[maxn], p[maxn], fa[maxn], ans = 0;

struct edge {
	int x, y; double w;
	edge() {}
	edge(int _, int __, double ___) {x = _; y = __; w = ___;}
}e[maxn];

int read() {
	int x = 0, f = 1; char ch = getchar();
	while(ch < '0' || ch > '9') {if(ch == '-') f = -1; ch = getchar();}
	while(ch >= '0' && ch <= '9') {x = x * 10 + ch - '0'; ch = getchar();}
	return f * x;
}

void getans() {
	int k, cnt = 0;
	while(1) {
		for(int i = 0; i <= n; i++) low[i] = 1e20;
		memset(fa, 0, sizeof(fa)); memset(p, 0, sizeof(p)); memset(vis, 0, sizeof(vis));
		for(int i = 1; i <= m; i++) if(e[i].x != e[i].y && e[i].w < low[e[i].y]) low[e[i].y] = e[i].w, fa[e[i].y] = e[i].x;
		low[rt] = 0; cnt = 0;
		for(int i = 1; i <= n; i++) {
			ans += low[i];
			for(k = i; (k != rt && vis[k] != i && !p[k]); k = fa[k]) vis[k] = i;
			if(vis[k] == i) {
				p[k] = ++cnt;
				for(int j = fa[k]; j != k; j = fa[j]) p[j] = cnt;
			}
		}
		if(!cnt) return;
		for(int i = 1; i <= n; i++) if(!p[i]) p[i] = ++cnt;
		for(int i = 1; i <= m; i++) {
			double t = low[e[i].y];
			e[i].x = p[e[i].x]; e[i].y = p[e[i].y];
			if(e[i].x != e[i].y) e[i].w -= t;
		}
		n = cnt; rt = p[rt];
	}
}

int main() {
	n = read(); int pts = 0;
	for(int i = 1; i <= n; i++) {
		scanf("%lf", &c[i]);
		g[i] = read();
		if(g[i]) {
			g[++pts] = g[i]; c[pts] = c[i];
			low[pts] = c[i]; id[i] = pts;
			e[pts] = edge(0, pts, c[pts]);
		}
	}
	rt = 0; n = pts; m = pts; q = read();
	while(q--) {
		int x = read(), y = read(); double t; scanf("%lf", &t);
		if(!id[x] || !id[y]) continue;
		e[++m] = edge(id[x], id[y], t); low[id[y]] = min(low[id[y]], t);
	}
	for(int i = 1; i <= n; i++) if(g[i] > 1) ans += low[i] * (g[i] - 1);
	getans();
	printf("%.2f\n", ans);
	return 0;
}
```
###网络流
####HLPP预留推进算法
```cpp
#include <bits/stdc++.h>

const int MAXN = 1203;
const int INF = INT_MAX;

struct Node {
    int v, f, index;

    Node(int v, int f, int index) : v(v), f(f), index(index) {}
};

std::vector<Node> edge[MAXN];
std::vector<int> list[MAXN], height, count, que, excess;
typedef std::list<int> List;
std::vector<List::iterator> iter;
List dlist[MAXN];
int highest, highestActive;
typedef std::vector<Node>::iterator Iterator;

inline void addEdge(const int u, const int v, const int f) {
    edge[u].push_back(Node(v, f, edge[v].size()));
    edge[v].push_back(Node(u, 0, edge[u].size() - 1));
}

inline void globalRelabel(int n, int t) {
    height.assign(n, n);
    height[t] = 0;
    count.assign(n, 0);
    que.clear();
    que.resize(n + 1);
    int qh = 0, qt = 0;
    for (que[qt++] = t; qh < qt;) {
        int u = que[qh++], h = height[u] + 1;
        for (Iterator p = edge[u].begin(); p != edge[u].end(); ++p) {
            if (height[p->v] == n && edge[p->v][p->index].f > 0) {
                count[height[p->v] = h]++;
                que[qt++] = p->v;
            }
        }
    }
    for (register int i = 0; i <= n; ++i) {
        list[i].clear();
        dlist[i].clear();
    }
    for (register int u = 0; u < n; ++u) {
        if (height[u] < n) {
            iter[u] = dlist[height[u]].insert(dlist[height[u]].begin(), u);
            if (excess[u] > 0) list[height[u]].push_back(u);
        }
    }
    highest = (highestActive = height[que[qt - 1]]);
}

inline void push(int u, Node &e) {
    int v = e.v;
    int df = std::min(excess[u], e.f);
    e.f -= df;
    edge[v][e.index].f += df;
    excess[u] -= df;
    excess[v] += df;
    if (0 < excess[v] && excess[v] <= df) list[height[v]].push_back(v);
}

inline void discharge(int n, int u) {
    int nh = n;
    for (Iterator p = edge[u].begin(); p != edge[u].end(); ++p) {
        if (p->f > 0) {
            if (height[u] == height[p->v] + 1) {
                push(u, *p);
                if (excess[u] == 0) return;
            } else {
                nh = std::min(nh, height[p->v] + 1);
            }
        }
    }
    int h = height[u];
    if (count[h] == 1) {
        for (register int i = h; i <= highest; ++i) {
            for (List::iterator it = dlist[i].begin(); it != dlist[i].end();
                 ++it) {
                count[height[*it]]--;
                height[*it] = n;
            }
            dlist[i].clear();
        }
        highest = h - 1;
    } else {
        count[h]--;
        iter[u] = dlist[h].erase(iter[u]);
        height[u] = nh;
        if (nh == n) return;
        count[nh]++;
        iter[u] = dlist[nh].insert(dlist[nh].begin(), u);
        highest = std::max(highest, highestActive = nh);
        list[nh].push_back(u);
    }
}

inline int hlpp(int n, int s, int t) {
    if (s == t) return 0;
    highestActive = 0;
    highest = 0;
    height.assign(n, 0);
    height[s] = n;
    iter.resize(n);
    for (register int i = 0; i < n; ++i)
        if (i != s)
            iter[i] = dlist[height[i]].insert(dlist[height[i]].begin(), i);
    count.assign(n, 0);
    count[0] = n - 1;
    excess.assign(n, 0);
    excess[s] = INF;
    excess[t] = -INF;
    for (register int i = 0; i < (int)edge[s].size(); ++i) push(s, edge[s][i]);
    globalRelabel(n, t);
    for (int u; highestActive >= 0;) {
        if (list[highestActive].empty()) {
            highestActive--;
            continue;
        }
        u = list[highestActive].back();
        list[highestActive].pop_back();
        discharge(n, u);
    }
    return excess[t] + INF;
}

inline int read() {
    int f = 0, fu = 1;
    char c = getchar();
    while(c < '0' || c > '9') {
        if(c == '-') fu = -1;
        c = getchar();
    }
    while(c >= '0' && c <= '9') {
        f = (f << 3) + (f << 1) + c - 48;
        c = getchar();
    }
    return f * fu;
}
int n, m, s, t,u,v,f;
int main() {
    n=read(),m=read(),s=read(),t=read();
    for (register int i = m;i>0; --i) {
        u = read(), v = read(), f = read();
        addEdge(u, v, f);
    }
    printf("%d",hlpp(n + 1, s, t)) ;
    return 0;
}
```
####最大流
```cpp
#define maxm 200005
#define maxn 10005
#define INF 1000000
#define getchar() (S==T&&(T=(S=BB)+fread(BB,1,1<<15,stdin),S==T)?EOF:*S++)
char BB[1<<15],*S=BB,*T=BB;
int ans, n, m, s, t, u, cur[maxn], d[maxn], v[maxm], c[maxm], pre[maxm], num[maxn], head[maxn], last[maxm], nf, cnp;
queue <int> q;
inline int read()
{
    int x=0;
    char c;
    c = getchar();
    while(c < '0' || c > '9') c = getchar();
    while(c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
    return x;
}
inline void add(int a,int b, int x)
{
    last[cnp] = head[a];
    v[cnp] = b, c[cnp] = x;
    head[a] = cnp++;
}
void BFS()
{
    q.push(t);
    d[t] = 1;
    while(!q.empty())
    {
        int u = q.front();
        ++num[d[u]];
        q.pop();
        for(int i = head[u]; i != -1; i = last[i])
        {
            int to = v[i]; 
            if(!d[to] && c[i ^ 1])
            {
                d[to] = d[u] + 1;
                q.push(to);
            }
        }
    }
}
inline int argu()
{
    while (u != s)
    {
        c[pre[u]] -= nf;
        c[pre[u] ^ 1] += nf;
        u = v[pre[u] ^ 1];
    }
    ans += nf;
}
int maxflow()
{
    u = s, nf = INF;
    bool done;
    memcpy(cur, head, sizeof(head));
    while(d[s] != n + 1)
    {
        if(u == t) argu(), nf = INF;
        done = false;
        for(int i = cur[u]; i != -1; i = last[i])
        {
            int to = v[i];
            if(c[i] && d[to] == d[u] - 1)
            {
                done = true;
                cur[u] = i;
                pre[to] = i;
                u = to;
                nf = min(nf, c[i]);
                break;
            }
        }
        if(!done)
        {
            int m = n + 1;
            for(int i = head[u]; i != -1; i = last[i])
            {
                int to = v[i];
                if(c[i]) m = min(m, d[to] + 1);
            }
            if(--num[d[u]] == 0) break;
            cur[u] = head[u];
            num[d[u] = m]++;
            if(u != s) u = v[pre[u] ^ 1];
        }
    }
    return ans;
}
int main()
{
    n = read(), m = read(), s = read(), t = read();
    memset(head,-1,sizeof(head));
    for(int i = 1; i <= m; ++i)
    {
        int u = read(), v = read(), w = read();
        add(u, v, w);
        add(v, u, 0);
    }
    BFS();
    printf("%d\n", maxflow());
    return 0;
}
```
####SPFA费用流
```cpp
#define ch_top 10000000
char ch[ch_top],*now_r=ch;
void read(int &x) 
{ while(*now_r<48)++now_r;
  for (x=*now_r-48;*++now_r>=48;)
   x= (x<<1)+(x<<3)+*now_r-48;
}

#define oo 1000000000
#define N 10100
#define M 200100
int n,m,S,T,i,x,y,f,c;
int t[N];
struct edge
{
    int to,next,f,cost;
}l[M];int e;
#define add(x,y,f,c) {l[++e]=(edge){y,t[x],f,c};t[x]=e;}

int g[N],tot,h[N];
int _t[N],base,now;
int rt,head[N],next[N],*dy[N];
#define fu(x,y) {dy[y]=&x;x=y;} 
#define _fu(x,y) {dy[y]=x;*x=y;} 
void merge(int &x,int y)
{
    if (g[x]<g[y])
    {
        fu(next[y],head[x]);fu(head[x],y);
    }else
    {
        fu(next[x],head[y]);fu(head[y],x);
        x=y;
    }
}
void merges(int &x)
{
    int y=next[x],r;
    while (y) 
    { 
       r=next[y];
       merge(x,y);
       y=r; 
    }
}
bool spfa()
{
    for (i=1;i<=n;++i) g[i]=oo;
    g[rt=T]=0;
    do
    {
        x=rt;
        merges(rt=head[rt]);
        dy[x]=0;next[x]=head[x]=0;
        base=g[x];
        for (_t[x]=i=t[x];i;i=l[i].next)
        if (l[i^1].f&&g[y=l[i].to]>(now=base-l[i].cost+h[x]-h[y]))
        {
            if(now>g[S]) continue; 
            g[y]=now;
            if (y!=rt)
            {
              if (dy[y]!=0) _fu(dy[y],next[*dy[y]])  else 
              if (!rt) {rt=y;continue;}
              merge(rt,y);
            }
        }
    }while (rt);
    
    if (g[S]==oo) return 0;
    for (x=1;x<=n;++x) h[x]+=g[x]; 
    return 1;
}

bool in[N];
int ansf=0,ansc=0;
int dfs(int x,int f)
{
    if (x==T) return f;
    in[x]=1;
    int f0=f,del,y;
    for (int &i=_t[x];i;i=l[i].next)
    if (l[i].f&&(!in[y=l[i].to])&&(l[i].cost==h[x]-h[y])) 
    {
        del=dfs(y,min(l[i].f,f));
        l[i].f-=del;l[i^1].f+=del;
        f-=del;
        if (!f) {in[x]=0;return f0;} 
    }
    in[x]=0;
    return f0-f;
}

int main()
{
    //freopen("1.in","r",stdin);
    fread(ch,1,ch_top,stdin);
    read(n);read(m);read(S);read(T);
    e=1; 
    for (i=1;i<=m;++i) 
    {
        read(x);read(y);read(f);read(c); 
        add(x,y,f,c) add(y,x,0,-c)
    }
    
    while (spfa()) 
    {
        f=dfs(S,oo);
        ansf+=f;ansc+=f*h[S];
    }
    printf("%d %d",ansf,ansc);
}
```
####zkw费用流
```cpp
#include <bits/stdc++.h>
using namespace std;
#define ll long long
#define N 5010
#define inf 0x3f3f3f3f

inline int read(){
	int x=0,f=1;char ch=getchar();
	while(ch<'0'||ch>'9'){if(ch=='-') f=-1;ch=getchar();}
	while(ch>='0'&&ch<='9') x=x*10+ch-'0',ch=getchar();
	return x*f;
}

int n,m,s,t,h[N],num=1,cost=0,price=0,mxflow=0;
bool vis[N];
struct edge{
	int to,next,w,c;
}data[50010<<1];
inline void add(int x,int y,int w,int c){
	data[++num].to=y;data[num].next=h[x];h[x]=num;data[num].w=w;data[num].c=c;
	data[++num].to=x;data[num].next=h[y];h[y]=num;data[num].w=0;data[num].c=-c;
}
inline int dinic(int x,int low){
	vis[x]=1;if(x==t){cost+=low*price;mxflow+=low;return low;}int tmp=low;
	for(int i=h[x];i;i=data[i].next){
		int y=data[i].to;if(vis[y]||!data[i].w||data[i].c) continue;
		int res=dinic(y,min(tmp,data[i].w));
		tmp-=res;data[i].w-=res;data[i^1].w+=res;
		if(!tmp) return low;
	}return low-tmp;
}
inline bool label(){
	int d=inf;
	for(int x=1;x<=n;++x){
		if(!vis[x]) continue;
		for(int i=h[x];i;i=data[i].next){
			int y=data[i].to;if(vis[y]) continue;
			if(data[i].w&&data[i].c<d) d=data[i].c;
		}
	}if(d==inf) return 0;
	for(int x=1;x<=n;++x){
		if(!vis[x]) continue;
		for(int i=h[x];i;i=data[i].next)
			data[i].c-=d,data[i^1].c+=d;
	}price+=d;return 1;
}

void mcmf()
{
	do 
		do
			memset(vis,0,sizeof(vis));
		while(dinic(s,inf));
	while(label());
} 

int main()
{
	n = read();
	m = read();
	s = read();
	t = read();
	while (m--)
	{
		int x = read(), y = read(), w = read(), c = read();
		add(x,y,w,c);
	}
	mcmf();
	printf("%d %d\n",mxflow,cost);
	return 0;
}
```
####上下界网络流
**无源汇网络可行流**
    无源汇网络是指在网络流图中没有明确指定源点和汇点，流在网络中是循环流动的，可行流是指网络中所有路径上的流量均满足 Flow(i) 属于[Bi, Ci]，且每个点的流入量之和等于流出量之和。 
    每条边必须有一个流量下界，这非常麻烦，考虑将流量下界单独出去，成为一个新图，使得边的流量下界为0，流量上界为 Ci-Bi，变成了一个普通的网络流问题（每条边只有流量上界，即容量）。称每条边的流量下界为必须流，每条边的流量减去流量下界为自由流，由于边的流量范围的限制，有些情况下网络流图可能无法流通。比如下图中国 1-->2 的边上的流量上界为3，而2-->3的边上的流量下界为4，那么网络就无法流动。网络中存在满足边的上下界要求的网络流，称为可行流。 
    求解可行流 
    设每个点所有流入量的流量下界之和IBi和所有流出量下界之和OBi，然后虚拟一个源点SS和一个汇点TT，使得原图中每个点都有OBi的流量流入SS，同时有IBi的流量从TT流入，这样，就将每个点的流量下界独立出来。这样，每条边的必须流就被SS和TT管理。 
    以SS，TT分别为新图的源点和汇点，寻找网络最大流，如果最大流使得从SS出发的每条路径都被填满（那么到达TT的每条路径也必然被填满），那么说明对于原图中的每个点，都满足最低流入量的流流入和最低流出量的流流出，从而存在满足原图流量下界的可行流。 
   加SS和TT之后的新图，从每个点的引出容量为最低流出量之和OBi的路径指向汇点TT，并从SS引入容量为最低流入量之和IBi的路径指向该点。 同时，每条边的最低流量变为0，容量变为Ci-Bi。 
    然后，从SS到TT寻找网络最大流。图中所示，SS-->1-->2-->TT的路径上流量为3，SS-->1-->TT路径上流量为1，SS-->2-->3-->TT路径上的流量为1，SS-->3-->TT路径上流量为3。则最大流为8，且能够使得SS到原图中每个点的边上的流量均满流。因此，原图存在可行流。
**有源汇网络的可行流**
    对于流量有上下界的有源汇网络，原图中存在源点S和汇点T，为了求可行流，先将其转换为无源汇网络。 
    从T-->S引入一条边，其流量上下界为[0,INF]，此时原图变为了无源汇网络，然后按照无源汇网络求解可行流的方法来求解。 
**有源汇网络的最大流**
    要求最大流，先求可行流，通过“有源汇网络的可行流”的求解方法来判断有源汇网络存在可行流。 
    若存在可行流，记从S流出的流量sum1，然后将T-->S的边取消，再次从S到T求解网络的最大流，记从S流出的流量sum2. 那么该有源汇网络的最大流为 sum1 + sum2. 
    其中，sum1是在网络满足流量下界的条件下，从源点S流出的流量；求出sum1之后，网络中可能还有余量可以继续增广，那么再次求解从S到T的最大流，得到sum2，sum1 + sum2即为最终的最大流。
**有源汇网络的最小流**
    求解有源汇网络最小流分为以下几步： 
（1）对SS到TT求一次最大流，即为f1.（在有源汇的情况下，先把整个网络趋向必须边尽量满足的情况）； 
（2）添加一条边T-->S，流量上限为INF，这条边即为P.(构造无源网络） 
（3）对SS到TT再次求最大流，即为f2。（判断此时的网络中存在可行流，同时求出SS-->TT最大流） 
    如果所有必须边都满流，证明存在可行解，原图的最小流为“流经边P的流量”（原图已经构造成无源汇网络，对于S同样满足 入流和==出流和，只有新添加的边流向S，而S的出流就是原图的最小流）。 
####一些经典建图
·搭配飞行员
题意
　　一群正驾驶，一群副驾驶。一些正驾驶可以和副驾驶一起飞。问最多多少架飞机可以飞。
题解
　　二分图最大匹配模型。
　　超级源向所有正驾驶连容量为1的边，所有副驾驶向超级汇连容量为1的边。可以一起飞的正副驾驶之间连容量为1的边。跑最大流就是二分图最大匹配。
·太空飞行计划
题意
　　m个实验，n个仪器。每个实验需要若干个仪器。激活每个仪器需要一个cost，每个实验有个价值。问激活哪些仪器收益最大。
题解
　　最大权闭合子图问题。可以转化为最小割问题。
　　超级源向所有实验连容量为收益边，仪器向超级汇连容量为cost的边。
　　如果实验i需要仪器j，则i向j连容量为INF的边。
　　则最大收益为收益和-最大流。
　　对应的解就是最小割划分出的S集合中的点，也就是最后一次增广找到阻塞流时能从S访问到的顶点。
　　定义一个割划分出的S集合为一个解，那么割集的容量之和就是(未被选的A集合中的顶点的权值 + 被选的B集合中的顶点的权值)，记为Cut。A集合中所有顶点的权值之和记为Total，那么Total - Cut就是(被选的A集合中的顶点的权值 - 被选的B集合中的顶点的权值)，即为我们的目标函数，记为A。要想最大化目标函数A，就要尽可能使Cut小，Total是固定值，所以目标函数A取得最大值的时候，Cut最小，即为最小割。
·最小路径覆盖
题意
　　图中的每个点都恰好在路径集合P中的一条路上，称集合P为一个路径覆盖。路径条数最少的覆盖称为最小路径覆盖。给出有向无环图G，求最小路径覆盖。
题解
　　拆点。讲i点拆为Xi和Yi。若i到j之间有边，则Xi向Yj连边。跑二分图最大匹配即可。
　　最小路径覆盖的条数=原图顶点数-二分图最大匹配数。
　　对于一个路径覆盖，有如下性质：
　　1、每个顶点属于且只属于一个路径。
　　2、路径上除终点外，从每个顶点出发只有一条边指向路径上的另一顶点。
　　所以我们可以把每个顶点理解成两个顶点，一个是出发，一个是目标，建立二分图模型。该二分图的任何一个匹配方案，都对应了一个路径覆盖方案。如果匹配数为0，那么显然路径数=顶点数。每增加一条匹配边，那么路径覆盖数就减少一个，所以路径数=顶点数 - 匹配数。要想使路径数最少，则应最大化匹配数，所以要求二分图的最大匹配。
·魔术球
题意
　　N根柱子，依次放编号为1，2，3..的球。要求每次只能在柱顶放球。要求相邻球编号和为完全平方数。
　　计算N根柱子最多放多少个球。
题解
　　设最多放X个球。那么建立节点1..X，若i<j且i+j为完全平方数，则建一条有向边。那么最小路径覆盖就是最少需要的柱子数。
　　顺序枚举A的值，当最小路径覆盖数刚好大于N时终止，A-1就是最优解。
　　若二分答案，则每次要重新建图，复杂度更高。
·圆桌聚餐
题意
　　n个单位，每个单位ri个人。共m张桌子，每个桌子可容纳ci个人。同桌的人不能同单位。
题解
　　二分图多重匹配。
　　超级源向所有单位连容量为单位人数的边，所有桌子向超级汇连容量为桌子容量的边。
　　所有单位向所有桌子连边。
　　跑最大流，若最大流=总人数，则有解，否则无解。
##数学
###快速GCD
```cpp
 inline long long gcd(long long a, long long b)
    {
        if (!a) return b;
        if (!b) return a;
        int t = __builtin_ctzll(a | b);
        a >>= __builtin_ctzll(a);
        do
        {
            b >>= __builtin_ctzll(b);
            if (a > b) swap(a, b);
            b -= a;
        } while (b);
        return a << t;
    }
```
###快速乘
```cpp
inline long long multi(long long x,long long y,long long mod)
{
long long tmp=(x*y-(long long)((long double)x/mod*y+1.0e-8)*mod);
return tmp<0 ? tmp+mod : tmp;
}
```
###慢速乘
```cpp
ll Slow_Mul(ll n, ll k, ll mod){
    ll ans = 0;
    while(k){
      if(k & 1) ans = (ans + n) % mod;
      k >>= 1;
      n = (n + n) % mod;
    }
    return ans;
}
```
###exCRT
```cpp
const int maxn=100010;
int n;
lt ai[maxn],bi[maxn];

lt mul(lt a,lt b,lt mod)
{
    lt res=0;
    while(b>0)
    {
        if(b&1) res=(res+a)%mod;
        a=(a+a)%mod;
        b>>=1;
    }
    return res;
}

lt exgcd(lt a,lt b,lt &x,lt &y)
{
    if(b==0){x=1;y=0;return a;}
    lt gcd=exgcd(b,a%b,x,y);
    lt tp=x;
    x=y; y=tp-a/b*y;
    return gcd;
}

lt excrt()
{
    lt x,y,k;
    lt M=bi[1],ans=ai[1];//第一个方程的解特判
    for(int i=2;i<=n;i++)
    {
        lt a=M,b=bi[i],c=(ai[i]-ans%b+b)%b;//ax≡c(mod b)
        lt gcd=exgcd(a,b,x,y),bg=b/gcd;
        if(c%gcd!=0) return -1; //判断是否无解，然而这题其实不用

        x=mul(x,c/gcd,bg);//把x转化为最小非负整数解
        ans+=x*M;//更新前k个方程组的答案
        M*=bg;
        ans=(ans%M+M)%M;
    }
    return (ans%M+M)%M;
}
```
###ex卢卡斯
```cpp
typedef long long ll;

ll exgcd(ll a,ll b,ll &x,ll &y)
{
    if(!b){x=1;y=0;return a;}
    ll res=exgcd(b,a%b,x,y),t;
    t=x;x=y;y=t-a/b*y;
    return res;
}

ll p;

inline ll power(ll a,ll b,ll mod)
{
    ll sm;
    for(sm=1;b;b>>=1,a=a*a%mod)if(b&1)
        sm=sm*a%mod;
    return sm;
}

ll fac(ll n,ll pi,ll pk)
{
    if(!n)return 1;
    ll res=1;
    for(register ll i=2;i<=pk;++i)
        if(i%pi)(res*=i)%=pk;
    res=power(res,n/pk,pk);
    for(register ll i=2;i<=n%pk;++i)
        if(i%pi)(res*=i)%=pk;
    return res*fac(n/pi,pi,pk)%pk;
}

inline ll inv(ll n,ll mod)
{
    ll x,y;
    exgcd(n,mod,x,y);
    return (x+=mod)>mod?x-mod:x;
}

inline ll CRT(ll b,ll mod){return b*inv(p/mod,mod)%p*(p/mod)%p;}

const int MAXN=11;

static ll n,m;

static ll w[MAXN];

inline ll C(ll n,ll m,ll pi,ll pk)
{
    ll up=fac(n,pi,pk),d1=fac(m,pi,pk),d2=fac(n-m,pi,pk);
    ll k=0;
    for(register ll i=n;i;i/=pi)k+=i/pi;
    for(register ll i=m;i;i/=pi)k-=i/pi;
    for(register ll i=n-m;i;i/=pi)k-=i/pi;
    return up*inv(d1,pk)%pk*inv(d2,pk)%pk*power(pi,k,pk)%pk;
}

inline ll exlucus(ll n,ll m)
{
    ll res=0,tmp=p,pk;
    static int lim=sqrt(p)+5;
    for(register int i=2;i<=lim;++i)if(tmp%i==0)
    {
        pk=1;while(tmp%i==0)pk*=i,tmp/=i;
        (res+=CRT(C(n,m,i,pk),pk))%=p;
    }
    if(tmp>1)(res+=CRT(C(n,m,tmp,tmp),tmp))%=p;
    return res;
}
```
###离散对数
```cpp
void bsgs(int u, int v) {
    if(!u && !v) return 2;
    else if(!u) return -1;
    int m = ceil(sqrt(mod)); hash.clear();
    hash[1] = m; int p = u, q = v, iv = inv(pow(u, m));
    for(int i = 1; i < m; i++, p = 1ll * p * u % mod) hash[p] = i;
    for(int i = 0; i <= m; i++) {
        if(hash[q]) return i * m + hash[q] % m + 1;
        q = 1ll * q * iv % mod;
    }
    return -1;
}
int exbsgs(int a, int b, int p) {
    a %= p; b %= p;
    if(b == 1) return 0; int cnt = 0; ll t = 1;
    if(a == 0) return b > 1 ? -1 : b == 0 && p > 1;
    for(int g = gcd(a, p); g != 1; g = gcd(a, p)) {
        if(b % g) return -1;
        p /= g; b /= g; t = 1ll * t * (1ll * a / g % p) % p;
        cnt++; if(b == t) return cnt;
    }
    hash.clear(); int m = ceil(sqrt(p));
    ll u = b; for(int i = 0; i < m; i++, u = 1ll * u * a % p) hash[u] = i;
    ll v = t, iv = pow(a, m, p);
    for(int i = 0; i <= m; i++) {
        v = 1ll * v * iv % p;
        if(hash.count(v)) return (i + 1) * m - hash[v] + cnt;
    }
    return -1;
}
```
###M-R Pho
```cpp
#include <bits/stdc++.h>
using namespace std;

typedef long long LL;

const int ps[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
const int pcnt = sizeof(ps) / sizeof(int);

inline LL quick_mul(LL a, LL b, LL m)
{
    LL d = ((long double)a / m * b + 1e-8);
    LL r = a * b - d * m;
    return r < 0 ? r + m : r;
}

inline LL quick_pow(LL a,LL b,LL m)
{
    LL res = 1;
    for(; b; b >>= 1, a = quick_mul(a, a, m))
        if (b & 1)
            res = quick_mul(res, a, m);
    return res;
}

inline LL gcd(LL a, LL b)
{
    if (!a || !b)
        return a + b;
    int t = __builtin_ctzll(a | b);
    a >>= __builtin_ctzll(a);
    do
    {
        b >>= __builtin_ctzll(b);
        if (a > b)
        {
            LL t = b;
            b = a;
            a = t;
        }
        b -= a;
    }
    while (b != 0);
    return a << t;
}

inline int Miller_Rabin(LL n)
{
    if (n == 1)
        return 0;
    if (n == 2 || n == 3 || n == 5)
        return 1;
    if (!(n & 1) || (n % 3 == 0) || (n % 5 == 0))
        return 0;
    LL m = n - 1;
    int k = 0;
    while (!(m & 1))
        m >>= 1, ++k;
    for (int i = 0; i < pcnt && ps[i] < n; ++i)
    {
        LL x = quick_pow(ps[i], m, n), y = x;
        for (int i = 0; i < k; ++i)
        {
            x = quick_mul(x, x, n);
            if (x == 1 && y != 1 && y != n - 1)
                return 0;
            y = x;
        }
        if (x != 1)
            return 0;
    }
    return 1;
}

inline LL next_rand(LL x, LL n, LL a)
{
    LL t = quick_mul(x, x, n) + a;
    return t < n ? t : t - n;
}

const int M = (1 << 7) - 1;
inline LL Pollard_Rho(LL n)
{
    if (n % 2 == 0)
        return 2;
    if (n % 3 == 0)
        return 3;
    LL x = 0, y = 0, t = 1, q = 1, a = (rand() % (n - 1)) + 1;
    for (int k = 2; true; k <<= 1, y = x, q = 1)
    {
        for (int i = 1; i < k; ++i)
        {
            x = next_rand(x, n, a);
            q = quick_mul(q, abs(x - y), n);
            if (!(i & M))
            {
                t = gcd(q, n);
                if (t > 1)
                    break;
            }
        }
        if (t > 1 || (t = gcd(q, n)) > 1)
            break;
    }
    if (t == n)
        for (t = 1; t == 1; t = gcd(abs((x = next_rand(x, n, a)) - y), n));
    return t;
}

LL f[105];
int cnt;

void solve(LL n)
{
    if (n == 1)
        return;
    if (Miller_Rabin(n))
    {
        f[cnt++] = n;
        return;
    }
    LL t = n;
    while (t == n)
        t = Pollard_Rho(n);
    solve(t);
    solve(n / t);
}

int main()
{
    LL n;
    while (~scanf("%lld", &n))
    {
        cnt = 0;
        solve(n);
        sort(f, f + cnt);
        for (int i = 0; i < cnt; ++i)
            printf("%lld ", f[i]);
        putchar('\n');
    }
    return 0;
}
```
###k次根
```cpp
ll gen(ll n, ll k)
{
    ll t = powl(n, 1.0 / k) - 0.5;
    return t + (powl(t + 1, k) - 0.5 <= n);
}
```
###逆元
```cpp
long long inv(long long t, long long p)
{
	return t == 1 ? 1 : (p - p / t) * inv(p % t, p) % p;
}

void ycl(int MOD)
{
    jc[0] = 1;
    for (int i = 1; i <= 2000; ++i)
    {
        jc[i] = jc[i - 1] * i;
        if (jc[i] >= MOD)
            jc[i] %= MOD;
    }
    inv[1] = 1;
    for (int i = 2; i <= 2000; ++i)  
        inv[i] = (LL)(MOD - MOD / i) * inv[MOD % i] % MOD; 
    inv[0] = 1;
    for (int i = 1; i <= 2000; ++i)  
        inv[i] = inv[i - 1] * inv[i] % MOD;
}
```
###杜教筛
设$S(n)=\sum_{i=1}^nf(i)$，$f(i)$是一个数论函数。
对任意数论函数$g(i)$，设$h=f*g$，则$\sum_{i=1}^nh(i)=\sum_{i=1}^{n}g(i)S(\lfloor \frac{n}{i} \rfloor)$
移项得$g(1)S(n)=\sum_{i=1}^n h(i)-\sum_{i=2}^ng(i)S(\lfloor \frac{n}{i} \rfloor)$
如果可以$O(\sqrt{n})$计算$\sum_{i=1}^n h(i)$，$O(1)$计算$g(i)$，那么可以快速递归求解，复杂度为$O(n^{\frac{3}{4}})$
如果$f(i)$是积性函数，则可以用欧拉筛求出前$n^{\frac{2}{3}}$项，再递归处理，复杂度为$O(n^{\frac{2}{3}})$
一些常见迪利克雷卷积
·$\sigma_0(n^2)=\sum_{d|n}2^{\omega(d)}$，其中$\sigma_k(n)$表示n的约数的k次幂之和，\omega(n)表示n的质因子个数。
·$\sum_{d|n}\phi(n)*\frac{n}{d}=n$
一些莫比乌斯恒等式
·$2^{\omega(n)}=\sum_{d|n}\mu^2(d)$
·$\mu^2(d)=\sum_{k^2|d}\mu(k)$
```cpp
const int maxn = 1700010;
int T, tot, prime[maxn], mu[maxn];
map<int, ll> ans_mu;

void sieve() {
    fill(prime, prime + maxn, 1);
    mu[1] = 1, tot = 0;
    for (int i = 2; i < maxn; i++) {
        if (prime[i]) {
            prime[++tot] = i, mu[i] = -1;
        }
        for (int j = 1; j <= tot && i * prime[j] < maxn; j++)         {
            prime[i * prime[j]] = 0;
            if (i % prime[j] == 0) {
                mu[i * prime[j]] = 0; break;
            } else {
                mu[i * prime[j]] = -mu[i];
            }
        }
    }
    for (int i = 2; i < maxn; i++) mu[i] += mu[i - 1];
}

ll calc_mu(int x) {
    if (x < maxn) return mu[x];
    if (ans_mu.count(x)) return ans_mu[x];
    ll ans = 1;
    for (ll i = 2, j; i <= x; i = j + 1) {
        j = x / (x / i), ans -= (j - i + 1) * calc_mu(x / i);
    }
    return ans_mu[x] = ans;
}

ll calc_phi(int x) {
    ll ans = 0;
    for (ll i = 1, j; i <= x; i = j + 1) {
        j = x / (x / i), ans += (x / i) * (x / i) * (calc_mu(j) - calc_mu(i - 1));
    }
    return ((ans - 1) >> 1) + 1;
}
```
###积分表
$\int kdx=kx+C$
$\int x^u=\frac{x^{u+1}}{u+1}+C$
$\int \frac{1}{x}dx=ln|x|+C$
$\int \frac{1}{1+x^2}dx=arctanx+C=-arccotx+C$
$\int \frac{1}{\sqrt{1-x^2}}dx=arcsinx+C=-arccosx+C$
$\int cosxdx=sinx+C$
$\int sinxdx=-cosx+C$
$\int \frac{1}{cos^2x}dx=tanx+C$
$\int \frac{1}{sin^2x}dx=-cotx+C$
$\int secxtanxdx=secx+C$
$\int cscxcotxdx=-cscx+C$
$\int e^xdx=e^x+C$
$\int a^xdx=\frac{a^x}{lna}+C$
$\int tanxdx=-ln|cosx|+C$
$\int cotxdx=ln|sinx|+C$
$\int secxdx=ln|secx+tanx|+C$
$\int cscxdx=ln|cscx-cotx|+C$
$\int \frac{1}{x^2+a^2}dx=\frac{1}{a}arctan\frac{x}{a}+C$
$\int \frac{1}{x^2-a^2}dx=\frac{1}{2a}ln|\frac{x-a}{x+a}|+C$
$\int \frac{1}{\sqrt{a^2-x^2}}dx=arcsin\frac{x}{a}+C$
$\int \frac{1}{\sqrt{x^2+a^2}}dx=ln|x+\sqrt{x^2+a^2}|+C$
$\int \frac{1}{\sqrt{x^2-a^2}}dx=ln|x+\sqrt{x^2-a^2}|+C$
###一些结论
####阿贝尔变换
$\sum_{i=m}^n f_k(g_{k+1}-g_k)=f_{n+1}g_{n+1}-f_mg_m-\sum_{k=m}^n g_{k+1}(f_{k+1}-f_k)$
####Freshman's Dream
$(x+y)^p=x^p+y^p$当x和y在特征为p的交换环中。例如模p意义下的加法。
$(\sum_{i=0}^{\infty}a_ix^i)^2\equiv \sum_{i=0}^{\infty}a_ix^2i(mod\ 2)$
####基本毕达哥拉斯三元组

$A= \begin{pmatrix}1&-2&2\\2&-1&2\\2&-2&3\end{pmatrix}$$B= \begin{pmatrix}1&2&2\\2&1&2\\2&2&3\end{pmatrix}$$C= \begin{pmatrix}-1&2&2\\-2&1&2\\-2&2&3\end{pmatrix}$

从(3,4,5)开始搜索即可
####伯努利数
$\sum_{k=0}^{n}{C_{n+1}^kB_k\ =\ 0} (B0 = 1)$
$\sum_{i\ =\ 1}^{n}{i^k\ =\ \frac{1}{k+1}\sum_{i=1}^{k+1}{C_{k+1}^iB_{k+1-i}{(n+1)}^i}} $
####特殊函数
对于一个x进制的数，每次把所有位加起来$(156_10=12_10=3_10)$，如果当前所有位数之和为y，y = 0, 则最后为0，y % (x – 1) = 0 && y > 0，则最后为x – 1，其他情况最后为y % (x – 1)
####四平方和定理+组合数相关结论
一个正整数必定能由4个平方数之和表示。
$\sum_{i=1}^{3}C_{n_i}^2  \sum_{i=1}^{4}C_{n_i}^3$  可以表示从1开始的连续很多个数
####欧拉定理
$F = E - V + 2$ 
####组合数技巧
$C_x^y=\sum_{i\ =\ 0}^{y}{C_n^iC_{x-n}^{y-i}}$  在y很小时，可以拆分运算
####Lindström–Gessel–Viennot lemma
网格图中n个起点到n个终点不相交的路径条数：
设n个起点分别为x1，x2……xn； n个终点分别为y1，y2……yn  (这些都是二维点)
方案数为
$det\begin{matrix}e(x1,\ y1)&\ldots\ldots&e(x1,\ yn)\\\ldots\ldots&\ldots\ldots&\ldots\ldots\\e(xn,\ y1)&\ldots\ldots&e(xn,\ yn)\\\end{matrix}​$
e(x1, y1)表示点x1到点y1的方案数，一般用C(n, m)表示

##代数算法
###FFT
```cpp
typedef struct Fs
{
    double x,y;
    Fs(){};
    Fs(double xx,double yy)
    {
        x=xx;y=yy;
    }
    Fs operator*(const Fs&a) const
    {
        return Fs(x*a.x-y*a.y,x*a.y+y*a.x);
    }
    Fs operator+(const Fs&a) const
    {
        return Fs(x+a.x,y+a.y);
    }
    Fs operator-(const Fs&a) const
    {
        return Fs(x-a.x,y-a.y);
    }
}Fs;
int f[1200000];
void FFT(Fs*a,int len,int on)
{
    int i,j,k;
    Fs t,w,Wn;
    for(i=0;i<len;i++)if(i<f[i])
    t=a[i],a[i]=a[f[i]],a[f[i]]=t;
    for(i=1;i<len;i<<=1)
    {
        Wn=Fs(cos(M_PI/i),on*sin(M_PI/i));
        for(j=0;j<len;j+=i<<1)
        {
            w=Fs(1,0);
            for(k=0;k<i;k++,w=w*Wn)
            {
                Fs x=a[j+k],y=a[j+i+k]*w;
                a[j+k]=x+y;
                a[j+i+k]=x-y;
            }
        }
    }
    if(on==-1)
        for(i=0;i<len;i++)a[i].x/=len;
}
int len=1,k=-1;
int N,M,i,j;
void ycl()
{
    while(len<N+M)len<<=1,k++;
    for(i=0;i<len;i++)f[i]=(f[i>>1]>>1)|((i&1)<<k);
}

//BZOJ4259 残缺的字符串
Fs A[1200000],B[1200000];
char s1[1000000],s2[1000000];
int ans[1000000],num;
char isint(double x)
{
    return x-(int)x<0.000001||x-(int)x>0.999999;
}
int main()
{
    scanf("%d%d%s%s",&N,&M,s1,s2);
    for(i=0;i<N;i++)
    {
        if(s1[i]=='*')A[N-i-1].x=0;
        else A[N-i-1].x=(1000+s1[i]);
    }
    for(i=0;i<M;i++)
    {
        if(s2[i]=='*')B[i].x=0;
        else B[i].x=1.0/(1000+s2[i]);
    }
    int len=1,k=-1;
    while(len<N+M)len<<=1,k++;
    for(i=0;i<len;i++)f[i]=(f[i>>1]>>1)|((i&1)<<k);
    FFT(A,len,1);FFT(B,len,1);
    for(i=0;i<len;i++)A[i]=A[i]*B[i];
    FFT(A,len,-1);
    for(i=N-1;i<M;i++)if(isint(A[i].x))ans[num++]=i-N+2;
    printf("%d\n",num);
    for(i=0;i<num;i++)printf("%d ",ans[i]);
}
```
###NTT
```cpp
const int MOD = 998244353;
const int N = 1048576;
const int M = 998244353;

int WM[N + 2], IWM[N + 2];
vector<int> IW[N + 2], W[N + 2];

int Pw(int a, int b, int p)
{
    int v = 1;
    for(; b; b >>= 1, a = 1ll * a * a % p)
        if (b & 1)
            v = 1ll * v * a % p;
    return v;
}

void ycl()
{
    for (int m = 2; m <= N; m <<= 1)
    {
        WM[m] = Pw(3, (M - 1) / m, M), IWM[m] = Pw(3, (M - 1) / m * (m - 1), M);
        int o = 1;
        W[m].push_back(o);
        for (int i = 1; i < m; ++i)
            o = 1ll * o * WM[m] % M, W[m].push_back(o);
        o = 1;
        IW[m].push_back(o);
        for (int i = 1; i < m; ++i)
            o = 1ll * o * IWM[m] % M, IW[m].push_back(o);
    }
}

void NTT(int *a, int n, int f = 1)
{
    int i, j, k, m, w, u, v;
    for (i = n >> 1, j = 1; j < n - 1; ++j)
    {
        if (i > j)
            swap(a[i], a[j]);
        for (k = n >> 1; k <= i; k >>= 1)
            i ^= k;
        i ^= k;
    }
    for (m = 2; m <= n; m <<= 1)
        for (i = 0; i < n; i += m)
            for (j = i; j < i + (m >> 1); ++j)
                if (a[j] || a[j + (m >> 1)])
                {
                    u = a[j];
                    v = 1ll * (f == 1 ? W[m][j - i] : IW[m][j - i]) * a[j + (m >> 1)] % M;
                    if ((a[j] = u + v) >= M)
                        a[j] -= M;
                    if ((a[j + (m >> 1)] = u - v) < 0)
                        a[j + (m >> 1)] += M;
                }
    if (f == -1)
        for (w = Pw(n, M - 2, M), i = 0; i < n; ++i)
            a[i] = 1ll * a[i] * w % M;
}

//BZOJ4259 残缺的字符串
char sa[N], sb[N], st[N];
int a[5][N], b[5][N], c[N], q[N];

int main()
{
    ycl();
    int m, n;
    scanf("%d%d", &m, &n);
    scanf("%s%s", sa, sb);
    for (int i = 0, j = m - 1; i < j; ++i, --j)
        swap(sa[i],sa[j]);
    for (int i = 0; i < m; ++i)
        a[1][i] = sa[i] - 'a' + 1;
    for (int i = 0; i < n; ++i)
        b[1][i] = sb[i] - 'a' + 1;
    for (int i = 0; i < m; ++i)
        if(sa[i] == '*')
            a[1][i] = 0;
    for (int i = 0; i < n; ++i)
        if(sb[i] == '*')
            b[1][i] = 0;
    int K;
    for (K = 1; K < n + m; K <<= 1);
    for (int i = 2; i <= 3; ++i)
    {
        for (int j = 0; j < m; ++j)
            a[i][j] = a[i - 1][j] * a[1][j];
        for (int j = 0; j < n; ++j)
            b[i][j] = b[i - 1][j] * b[1][j];
    }
    for (int i = 1; i < 4; ++ i)
    {
        NTT(a[i], K);
        NTT(b[i], K);
    }
    for (int i = 0; i < K; ++ i)
    {
        c[i] = (c[i] + 1LL * a[3][i] * b[1][i] % MOD) % MOD;
        c[i] = (c[i] + MOD - 2LL * a[2][i] * b[2][i] % MOD) % MOD;
        c[i] = (c[i] + 1LL * a[1][i] * b[3][i] % MOD) % MOD;
    }
    NTT(c, K, -1);
    
    int ans(0);
    for (int i = m - 1; i < n; ++i)
        if (!c[i])
            q[++ans] = i - m + 2;
    printf("%d\n", ans);
    for (int i = 1; i < ans; ++i)
        printf("%d ", q[i]);
    if (ans)
        printf("%d\n", q[ans]);

    return 0;
}
```
###FWT
```cpp
const int MAXN=(1<<17)+5;

typedef long long ll;

static int n,Len=1<<17;

static int S[MAXN],Fi[MAXN],bit[MAXN];

const int mod=998244353;

inline int ad(int u,int v){return (u+=v)>=mod?u-mod:u;}

inline void predone(int lim)
{
    Fi[0]=0;Fi[1]=1;
    Rep(i,1,lim)bit[i]=bit[i>>1]+(i&1);
}

inline void FMT(int *a)//快速莫比乌斯变换 FWTor
{
    for(register int z=1;z<Len;z<<=1)
        Rep(j,0,Len-1)if(z&j)a[j]=ad(a[j],a[j^z]);
}

inline void IFMT(int *a)//快速莫比乌斯反演
{
    for(register int z=1;z<Len;z<<=1)
        Rep(j,0,Len-1)if(z&j)a[j]=ad(a[j],mod-a[j^z]);
}

inline void FWTand(int *a)//快速沃尔什变换及其反演
{
    for(register int i=2,ii=1;i<=Len;i<<=1,ii<<=1)
        for(register int j=0;j<Len;j+=i)
            for(register int k=0;k<ii;++k)
                a[j+k]=ad(a[j+k],a[j+k+ii]);
}

inline void IFWTand(int *a)
{
    for(register int i=2,ii=1;i<=Len;i<<=1,ii<<=1)
        for(register int j=0;j<Len;j+=i)
            for(register int k=0;k<ii;++k)
                a[j+k]=ad(a[j+k],mod-a[j+k+ii]);
}

inline void FWTxor(int *a)
{
    static int t;
    for(register int i=2,ii=1;i<=Len;i<<=1,ii<<=1)
        for(register int j=0;j<Len;j+=i)
            for(register int k=0;k<ii;++k)
            {
                t=a[j+k];
                a[j+k]=ad(t,a[j+k+ii]);
                a[j+k+ii]=ad(t,mod-a[j+k+ii]);
            }
}

inline int div2(int x){return x&1?(mod+x)/2:x/2;}

inline void IFWTxor(int *a)
{
    static int t;
    for(register int i=2,ii=1;i<=Len;i<<=1,ii<<=1)
        for(register int j=0;j<Len;j+=i)
            for(register int k=0;k<ii;++k)
            {
                t=a[j+k];
                a[j+k]=div2(ad(t,a[j+k+ii]));
                a[j+k+ii]=div2(ad(t,mod-a[j+k+ii]));
            }
}

static int A[MAXN],B[MAXN],C[MAXN];

inline void solve()
{
    FMT(A);FMT(B);
    Rep(i,0,Len-1)C[i]=(unsigned long long)A[i]*B[i]%mod;
    IFMT(A);IFMT(B);IFMT(C);
    Rep(i,0,Len-1)write(C[i],' ');
    putchar('\n');
    FWTand(A);FWTand(B);
    Rep(i,0,Len-1)C[i]=(unsigned long long)A[i]*B[i]%mod;
    IFWTand(A);IFWTand(B);IFWTand(C);
    Rep(i,0,Len-1)write(C[i],' ');
    putchar('\n');
    FWTxor(A);FWTxor(B);
    Rep(i,0,Len-1)C[i]=(unsigned long long)A[i]*B[i]%mod;
    IFWTxor(C);
    Rep(i,0,Len-1)write(C[i],' ');
    putchar('\n');
}
```
###高斯消元
```cpp
bool gauss() {
	int cur = 1, nxt; double pivot;
	for(int i = 1; i <= n; i++) {
		for(nxt = cur; nxt <= n; nxt++) if(fabs(a[nxt][i]) > eps) break;
		if(nxt > n) continue;
		if(nxt != cur) for(int j = 1; j <= n + 1; j++) swap(a[nxt][j], a[cur][j]);
		pivot = a[cur][i]; for(int j = 1; j <= n + 1; j++) a[cur][j] /= pivot;
		for(int j = 1; j <= n; j++) if(j != cur) {
			pivot = a[j][i]; for(int k = 1; k <= n + 1; k++) a[j][k] -= pivot * a[cur][k];
		}
		cur++;
	}
	for(int i = cur; i <= n; i++) if(fabs(a[i][n + 1]) > eps) return 0;
	return 1;
}
```
### 线性基

```c++
struct linear_base {
    int L, bas[32];
    void Clear() {L = 0; memset(bas, 0, sizeof(bas));}
    void Insert(int x) {
        for(int i = 1; i <= L; ++i) x = min(x, x ^ bas[i]);
        if(!x) return;
  		bas[++L] = x;
        for(int i = L; i > 1; --i) if(bas[i] > bas[i - 1]) swap(bas[i], bas[i - 1]);
    }
    void Merge(const linear_base &y) {
        for(int i = 1; i <= y.L; ++i) this -> Insert(y.bas[i]);
    }
    int Get_max() {
        int x = 0;
        for(int i = 1; i <= L; ++i) x = max(x, x ^ bas[i]);
        return x;
    }
}
```
###pell方程
求$x^2-dy^2=1$第k大解
$x_n=x_{n-1}x_1+dy_{n-1}y_1$
$y_n=x_{n-1}y_1+y_{n-1}x_1$
则$\begin{pmatrix}x_k\\y_k\end{pmatrix}=\begin{pmatrix}x_1&dy_1\\y_1&x_1\end{pmatrix}^{k-1}\begin{pmatrix}x_1\\y_1\end{pmatrix}$
```cpp
struct matrix{ll a[2][2];matrix(){memset(a,0,sizeof(a));}};
matrix ans;
matrix multi(matrix a,matrix b)
{
    matrix ans;
    for(int i=0;i<2;i++)
        for(int j=0;j<2;j++)
            for(int k=0;k<2;k++)
                ans.a[i][j]=(ans.a[i][j]+a.a[i][k]*b.a[k][j]%mod)%mod;
    return ans;
}
matrix qpow(matrix res,ll k)
{
    while(k)
    {
        if(k&1)
            res=multi(res,ans);
        k/=2;
        ans=multi(ans,ans);
    }
    return res;
}
int main()
{
    ll n,k;
    while(scanf("%lld%lld",&n,&k)!=EOF)
    {
        ll nn=sqrt(n),keepx,keepy;
        if(nn*nn==n){printf("No answers can meet such conditions\n");continue;}
        for(int i=1;;i++){ll y=i*i*n+1;ll yy=sqrt(y);if(yy*yy==i*i*n+1){keepy=i;keepx=yy;break;}}
        ans.a[0][0]=keepx%mod;
        ans.a[0][1]=n*keepy%mod;
        ans.a[1][0]=keepy%mod;
        ans.a[1][1]=keepx%mod;
        matrix res;
        res.a[0][0]=1,res.a[1][1]=1;
        matrix ans=qpow(res,k-1);
        printf("%lld\n",(ans.a[0][0]*keepx%mod+ans.a[0][1]*keepy%mod+mod)%mod);
    }
    return 0;
}
```
##计算几何
### 二维几何

```c++
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <set>
#include <vector>
#include <queue>
#include <cmath>
#include <cstring>
#include <ctime>
#include <map>
#define mp make_pair
#define pub push_back
#define pob pop_back
#define pof pop_front
#define pii pair<int, int>
#define fi first
#define se second
#define MOD 1000000007
#define MOD2 998244353
#define LL long long
#define ULL unsigned long long
#define ui unsigned int
#define bas 26
#define bas2 131
using namespace std;
const int N = 200005;
const int M = 8000005;
const int INF = 1000000000;
typedef double db;
const db eps = 1e-8;
const db pi = acos(-1);
const db inf = 1e30;
inline int sign(db a) { return a < -eps ? -1 : a > eps;}
inline int cmp(db a, db b) {return sign(a - b);}
inline int inmid(db a, db b, db c) {return sign(a - c) * sign(b - c) <= 0;}//c在[a,b]内
//L line S segment P point
struct point {
    db x, y;
    point() {}
    point(db _x, db _y): x(_x), y(_y) {}
    point operator + (const point &p) const {return point(x + p.x, y + p.y);}
    point operator - (const point &p) const {return point(x - p.x, y - p.y);}
    point operator * (db k) const {return point(x * k, y * k);}
    point operator / (db k) const {return point(x / k, y / k);}
    int operator == (const point &p) const {return cmp(x, p.x) == 0 && cmp(y, p.y) == 0;}
    point turn(db k) {return point(x * cos(k) - y * sin(k), x * sin(k) + y * cos(k));}
    point turn90() {return point(-y, x);}
    bool operator < (const point &p) const {
        int c = cmp(x, p.x);
        if(c) return c == -1;
        return cmp(y, p.y) == -1;
    }
    bool operator > (const point &p) const {
        int c = cmp(x, p.x);
        if(c) return c == 1;
        return cmp(y, p.y) == 1;
    }
    db abs() {return sqrt(x * x + y * y);}
    db abs2() {return x * x + y * y;}
    db disto(point p) {return (*this - p).abs();}
    db alpha() {return atan2(y, x);}
    point unit() {db z = abs(); return point(x / z, y / z);}
    void scan() {db _x, _y; scanf("%lf%lf", &_x, &_y); x = _x, y = _y;}
    void print() {printf("%.11lf %.11lf\n", x, y);}
    point getdel() {return (sign(x) == -1 || (sign(x) == 0 && sign(y) == -1)) ? (*this) * (-1) : (*this);}
    int getP() const{return sign(y) == 1 || (!sign(y) && sign(x) == -1);}//向量相对方向
};
int inmid(point p1, point p2, point p3) {return inmid(p1.x, p2.x, p3.x) && inmid(p1.y, p2.y, p3.y);}
db dot(point p1, point p2) {return p1.x * p2.x + p1.y * p2.y;}
db cross(point p1, point p2) {return p1.x * p2.y - p1.y * p2.x;}
db rad(point p1, point p2) {return atan2(cross(p1, p2), dot(p1, p2));}
int compareangle(point p1, point p2) {
    return p1.getP() < p2.getP() || (p1.getP() == p2.getP() && sign(cross(p1, p2)) > 0);
}
point proj(point p1, point p2, point q) {//q到直线p1, p2的投影
    point p = p2 - p1;
    return p1 + p * (dot(q - p1, p) / p.abs2());
}
point reflect(point p1, point p2, point q) {return proj(p1, p2, q) * 2 - q;}
int clockwise(point p1, point p2, point p3) {//p1, p2, p3 逆时针 1 顺时针 -1 其他 0
    return sign(cross(p2 - p1, p3 - p1));
}
int checkLL(point p1, point p2, point p3, point p4) {//求直线p1, p2 和直线p3, p4是否有交点
    return cmp(cross(p3 - p1, p4 - p1), cross(p3 - p2, p4 - p2)) != 0;
}
point getLL(point p1, point p2, point p3, point p4) {//求直线p1, p2 和直线p3, p4的交点
    db w1 = cross(p1 - p3, p4 - p3), w2 = cross(p4 - p3, p2 - p3);
    return (p1 * w2 + p2 * w1) / (w1 + w2);
}
int intersect(db l1, db r1, db l2, db r2) {//判断两个区间(l1, r1)和(l2, r2)是否有交
    if(l1 > r1) swap(l1, r1);
    if(l2 > r2) swap(l2, r2);
    return cmp(r1, l2) != -1 && cmp(r2, l1) != -1;
}
int checkSS(point p1, point p2, point p3, point p4) {//判断线段p1, p2和线段p3, p4是否有交点
    return intersect(p1.x, p2.x, p3.x, p4.x) && intersect(p1.y, p2.y, p3.y, p4.y) &&
    sign(cross(p3 - p1, p4 - p1)) * sign(cross(p3 - p2, p4 - p2)) <= 0 &&
    sign(cross(p1 - p3, p2 - p3)) * sign(cross(p1 - p4, p2 - p4)) <= 0;
}
db disSP(point p1, point p2, point q) {//点到线段的距离
    point p3 = proj(p1, p2, q);
    return inmid(p1, p2, p3) ? q.disto(p3) : min(q.disto(p1), q.disto(p2));
}
db disSS(point p1, point p2, point p3, point p4) {
    if(checkSS(p1, p2, p3, p4)) return 0;
    return min(min(disSP(p1, p2, p3), disSP(p1, p2, p4)), min(disSP(p3, p4, p1), disSP(p3, p4, p2)));
}
int onS(point p1, point p2, point q) {
    return inmid(p1, p2, q) && sign(cross(p1 - q, p2 - q)) == 0;
}
struct circle {
    point o; db r;
    circle(point _o, db _r): o(_o), r(_r) {}
    void scan() {o.scan(); scanf("%lf", &r);}
    int inside(point p) {return cmp(r, o.disto(p));}
};
struct line {//ps[0] -> ps[1] 有向直线
    point ps[2];
    line(point p1, point p2) {ps[0] = p1, ps[1] = p2;}
    point& operator[] (int k) {return ps[k];}
    int include(point p) {return sign(cross(ps[1] - ps[0], p - ps[0])) > 0;}//在这个半平面内
    point dir() {return ps[1] - ps[0];}
    line push() {//向外平移eps
        point delta = (ps[1] - ps[0]).turn90().unit() * eps;
        return line(ps[0] - delta, ps[1] - delta);
    }
};
point getLL(line l1, line l2) {return getLL(l1[0], l1[1], l2[0], l2[1]);}
int checkLL(line l1, line l2) {return checkLL(l1[0], l1[1], l2[0], l2[1]);}
int parallel(line l1, line l2) {return sign(cross(l1.dir(), l2.dir())) == 0;}
int samedir(line l1, line l2) {return parallel(l1, l2) && sign(dot(l1.dir(), l2.dir())) == 1;}
int operator < (line l1, line l2) {
    if(samedir(l1, l2)) return l2.include(l1[0]);
    return compareangle(l1.dir(), l2.dir());
}
db area(vector<point> ps) {
    db res = 0;
    for(int i = 0, l = (int)ps.size(); i < l; ++i) res += cross(ps[i], ps[(i + 1) % l]);
    return fabs(res / 2);
}
int contain(vector<point> ps, point p) {//2:内部 1:边界 0:外部
    int n = (int)ps.size(), res = 0;
    for(int i = 0; i < n; ++i) {
        point p1 = ps[i], p2 = ps[(i + 1) % n];
        if(onS(p1, p2, p)) return 1;
        if(cmp(p1.y, p2.y) > 0) swap(p1, p2);
        if(cmp(p1.y, p.y) >= 0 || cmp(p2.y, p.y) < 0) continue;
        if(sign(cross(p1 - p2, p - p2)) < 0) res ^= 1;
    }
    return res << 1;
}
vector<point> convex(vector<point> ps) {
    int n = ps.size(); if(n <= 1) return ps;
    sort(ps.begin(), ps.end());
    vector<point> qs(n * 2); int k = 0;
    for(int i = 0; i < n; qs[k++] = ps[i++]) {
        while(k > 1 && sign(cross(qs[k - 1] - qs[k - 2], ps[i] - qs[k - 2])) <= 0) --k;
    }
    for(int i = n - 2, t = k; i >= 0; qs[k++] = ps[i--]) {
        while(k > t && sign(cross(qs[k - 1] - qs[k - 2], ps[i] - qs[k - 2])) <= 0) --k;
    }
    qs.resize(k - 1);
    return qs;
}
vector<point> convexNonStrict(vector<point> ps) {
    //需要所有点都是独一的 结果会把边界上的点都算进去
    int n = ps.size(); if(n <= 1) return ps;
    sort(ps.begin(), ps.end());
    vector<point> qs(n * 2); int k = 0;
    for(int i = 0; i < n; qs[k++] = ps[i++]) {
        while(k > 1 && sign(cross(qs[k - 1] - qs[k - 2], ps[i] - qs[k - 2])) < 0) --k;
    }
    for(int i = n - 2, t = k; i >= 0; qs[k++] = ps[i--]) {
        while(k > t && sign(cross(qs[k - 1] - qs[k - 2], ps[i] - qs[k - 2])) < 0) --k;
    }
    qs.resize(k - 1);
    return qs;
}
db convexDiameter(vector<point> ps) {
    int n = (int)ps.size(); if(n <= 1) return 0;
    int is = 0, js = 0;
    for(int k = 1; k < n; ++k) {
        is = ps[k] < ps[is] ? k : is;
        js = ps[js] < ps[k] ? k : js;
    }
    int i = is, j = js;
    db res = ps[i].disto(ps[j]);
    do {
        if(cross(ps[(i + 1) % n] - ps[i], ps[(j + 1) % n] - ps[j]) >= 0) j = (j + 1) % n;
        else i = (i + 1) % n;
        res = max(res, ps[i].disto(ps[j]));
    }while(i != is || j != js);
    return res;
}
vector<point> convexCut(const vector<point> &ps, point p1, point p2) {
    vector<point> qs;//凸包与这条直线必须同向
    int n = (int)ps.size();
    for(int i = 0; i < n; ++i) {
        point p3 = ps[i], p4 = ps[(i + 1) % n];
        int d1 = sign(cross(p2 - p1, p3 - p1)), d2 = sign(cross(p2 - p1, p4 - p1));
        if(d1 >= 0) qs.pub(p3);
        if(d1 * d2 < 0) qs.pub(getLL(p1, p2, p3, p4));
    }
    return qs;
}
bool cmpy(const point &lhs, const point &rhs) {return lhs.y < rhs.y;}
db closepoint(vector<point> &ps, int l, int r) {//最近点对，先要按照x排序 //还可以优化成nlogn
    if(r - l <= 5) {
        db res = inf;
        for(int i = l; i < r; ++i) {
            for(int j = i + 1; j <= r; ++j) res = min(res, ps[i].disto(ps[j]));
        }
        return res;
    }
    int mid = (l + r) >> 1, tx = ps[mid].x;
    db res = min(closepoint(ps, l, mid), closepoint(ps, mid + 1, r));
    vector<point> qs;
    for(int i = l; i <= r; ++i) if(abs(ps[i].x - tx) <= res) qs.pub(ps[i]);
    sort(qs.begin(), qs.end(), cmpy); int len = (int)qs.size();
    for(int i = 0; i < len; ++i) {
        for(int j = i + 1; j < len && qs[j].y - qs[i].y; ++j) res = min(res, qs[i].disto(qs[j]));
    }
    return res;
}
int checkpos(line l1, line l2, line l3) {return l3.include(getLL(l1, l2));}
vector<point> getHalfPlane(vector<line> &L) {//求半平面交，半平面是逆时针方向，输出按逆时针
    sort(L.begin(), L.end()); deque<line> q;
    for(int i = 0, il = (int)L.size(); i < il; ++i) {
        if(i && samedir(L[i], L[i - 1])) continue;
        while(q.size() > 1 && !checkpos(q[q.size() - 2], q[q.size() - 1], L[i])) q.pob();
        while(q.size() > 1 && !checkpos(q[0], q[1], L[i])) q.pof();
        q.pub(L[i]);
    }
    while(q.size() > 2 && !checkpos(q[q.size() - 2], q[q.size() - 1], q[0])) q.pob();
    while(q.size() > 2 && !checkpos(q[1], q[0], q[q.size() - 1])) q.pof();
    vector<point> res;
    for(int i = 0, l = (int)q.size(); i < l; ++i) res.pub(getLL(q[i], q[(i + 1) % l]));
    return res;
}
int checkposCC(circle c1, circle c2) { //返回两个圆公切线数量
    if(cmp(c1.r, c2.r) == -1) swap(c1, c2);
    db dis = c1.o.disto(c2.o);
    int w1 = cmp(dis, c1.r + c2.r), w2 = cmp(dis, c1.r - c2.r);
    if(w1 > 0) return 4; else if(!w1) return 3; else if(w2 > 0) return 2;
    else if(!w2) return 1; else return 0;
}
vector<point> getCL(circle c, point p1, point p2) {//沿着p1 -> p2方向給出，相切給出两个
    point p = proj(p1, p2, c.o); db d = c.r * c.r - (p - c.o).abs2();
    if(sign(d) == -1) return {};
    point delta = (p2 - p1).unit() * sqrt(max(0.0, d));
    return {p - delta, p + delta};
}
vector<point> getCC(circle c1, circle c2) { //沿c1逆时针給出，相切給出两个
    int pd = checkposCC(c1, c2); if(pd == 0 || pd == 4) return {};
    db a = (c1.o - c2.o).abs2(), cosA = (c1.r * c1.r + a - c2.r * c2.r) / (2 * c1.r * sqrt(max(0.0, a)));
    db b = c1.r * cosA, c = sqrt(max(0.0, c1.r * c1.r - b * b));
    point t = (c2.o - c1.o).unit(), p = c1.o + t * b, delta = t.turn90() * c;
    return {p - delta, p + delta};
}
vector<point> tangentCP(circle c1, point p1) {//沿c逆时针给出两个切点
    db a = (p1 - c1.o).abs(), b = c1.r * c1.r / a, c = sqrt(max(0.0, c1.r * c1.r - b * b));
    point t = (p1 - c1.o).unit(), mid = c1.o + t * b, delta = t.turn90() * c;
    return {mid - delta, mid + delta};
}
vector<line> tangentoutCC(circle c1, circle c2) {//只有外切线
    int pd = checkposCC(c1, c2); if(!pd) return {};
    if(pd == 1) {point p = getCC(c1, c2)[0]; return {line(p, p)};}
    if(cmp(c1.r, c2.r) == 0) {
        point delta = (c2.o - c1.o).unit().turn90().getdel();
        return {line(c1.o - delta * c1.r, c2.o - delta * c2.r), line(c1.o + delta * c1.r, c2.o + delta * c2.r)};
    }
    else {
        point p = (c1.o * c2.r - c2.o * c1.r) / (c2.r - c2.r);
        vector<point> A = tangentCP(c1, p), B = tangentCP(c2, p);
        vector<line> res;
        for(int i = 0, l = (int)A.size(); i < l; ++i) res.pub(line(A[i], B[i]));
        return res;
    }
}
vector<line> tangentinCC(circle c1, circle c2) {
    int pd = checkposCC(c1, c2); if(pd <= 2) return {};
    if(pd == 3) {point p = getCC(c1, c2)[0]; return {line(p, p)};}
    point p = (c1.o * c2.r + c2.o * c1.r) / (c1.r + c2.r);
    vector<point> A = tangentCP(c1, p), B = tangentCP(c2, p);
    vector<line> res;
    for(int i = 0, l = (int)A.size(); i < l; ++i) res.pub(line(A[i], B[i]));
    return res;
}
vector<line> tangentCC(circle c1, circle c2) {
    int flag = 0; if(c1.r < c2.r) swap(c1, c2), flag = 1;
    vector<line> A = tangentoutCC(c1, c2), B = tangentinCC(c1, c2);
    for(line k: B) A.pub(k);
    if(flag) for(line &k: A) swap(k[0], k[1]);
    return A;
}
db areaCT(circle c1, point p2, point p3) {//圆c1和三角形p1, p2, c1.o的有向面积交
    point p1 = c1.o; c1.o = c1.o - p1; p2 = p2 - p1; p3 = p3 - p1;
    int pd1 = c1.inside(p2), pd2 = c1.inside(p3);
    vector<point> A = getCL(c1, p2, p3);
    if(pd1 >= 0) {
        if(pd2 >= 0) return cross(p2, p3) / 2;
        else return c1.r * c1.r * rad(A[1], p3) / 2 + cross(p2, A[1]) / 2;
    }
    else if(pd2 >= 0) return c1.r * c1.r * rad(p2, A[0]) / 2 + cross(A[0], p3) / 2;
    else {
        int pd = cmp(c1.r, disSP(p2, p3, c1.o));
        if(pd <= 0) return c1.r * c1.r * rad(p2, p3) / 2;
        else return c1.r * c1.r * (rad(p2, A[0]) + rad(A[1], p3)) / 2 + cross(A[0], A[1]) / 2;
    }
}
circle getcircle(point p1, point p2, point p3) {//返回3个点的外接圆
    db a1 = p2.x - p1.x, b1 = p2.y - p1.y, c1 = (a1 * a1 + b1 * b1) / 2;
    db a2 = p3.x - p1.x, b2 = p3.y - p1.y, c2 = (a2 * a2 + b2 * b2) / 2;
    db d = a1 * b2 - a2 * b1;
    point o = point(p1.x + (c1 * b2 - c2 * b1) / d, p1.y + (c2 * a1 - c1 * a2) / d);
    return circle(o, p1.disto(o));
}
circle getScircle(vector<point> ps) {
    random_shuffle(ps.begin(), ps.end());
    circle res = circle(ps[0], 0);
    int n = (int)ps.size();
    for(int i = 1; i < n; ++i) if(res.inside(ps[i]) == -1) {
        res = circle(ps[i], 0);
        for(int j = 0; j < i; ++j) if(res.inside(ps[j]) == -1) {
            res.o = (ps[i] + ps[j]) / 2; res.r = res.o.disto(ps[i]);
            for(int k = 0; k < j; ++k) {
                if(res.inside(ps[k]) == -1) res = getcircle(ps[i], ps[j], ps[k]);
            }
        }
    }
    return res;
}
typedef pair<db, int> pdi;
point a[N];
pdi alp[N << 2];
bool cmp_alp(const pdi &lhs, const pdi &rhs) {
    if(sign(lhs.fi - rhs.fi) == 0) return lhs.se < rhs.se;
    else return lhs.fi < rhs.fi;
}
bool check(db R, int n, int s) {//判断半径为R的圆能否覆盖s个点，最后的res+1位最多可覆盖的点数
    int res = 0;
    for(int i = 1; i <= n; ++i) {
        int sum = 0, tot = 0;
        for(int j = 1; j <= n; ++j) {
            db dis = a[i].disto(a[j]);
            if(i == j || sign(dis - R * 2) > 0) continue;
            db theta = atan2(a[j].y - a[i].y, a[j].x - a[i].x);
            if(theta < 0) theta += pi * 2;
            db phi = acos(dis / (R * 2));
            db le = theta - phi, ri = theta + phi;
            if(sign(le) >= 0 && sign(ri - 2 * pi) <= 0) {
                alp[++tot] = mp(le, 1);
                alp[++tot] = mp(ri, -1);
            }
            else if(sign(le) < 0) {
                alp[++tot] = mp(0, 1);
                alp[++tot] = mp(ri, -1);
                alp[++tot] = mp(le + 2 * pi, 1);
                alp[++tot] = mp(2 * pi, -1);
            }
            else {
                alp[++tot] = mp(0, 1);
                alp[++tot] = mp(ri - 2 * pi, -1);
                alp[++tot] = mp(le, 1);
                alp[++tot] = mp(2 * pi, -1);
            }
        }
        sort(alp + 1, alp + tot + 1, cmp_alp);
        for(int i = 1; i <= tot; ++i) {
            sum += alp[i].se;
            res = max(res, sum);
        }
    }
    return res + 1 >= s;
}
struct CH {
    int n;
    vector<point> ps, lower, upper;
    point operator[] (int i) {return ps[i];}

    int Find(vector<point> &vec, point dir) {
        int l = 0, r = vec.size();
        while(l + 5 < r) {
            int L = (l * 2 + r) / 3, R = (l + r * 2) / 3;
            if(dot(vec[L], dir) > dot(vec[R], dir)) r = R;
            else l = L;
        }
        int res = l;
        for(int i = l + 1; i < r; ++i) if(dot(vec[i], dir) > dot(vec[res], dir)) res = i;
        return res;
    }
    void init(vector<point> _ps) {
        ps = _ps, n = ps.size();
        rotate(ps.begin(), min_element(ps.begin(), ps.end()), ps.end());
        int id = max_element(ps.begin(), ps.end()) - ps.begin();
        lower = vector<point>(ps.begin(), ps.begin() + id + 1);
        upper = vector<point>(ps.begin() + id, ps.end()); upper.pub(ps[0]);
    }
    int findfarest(point dir) {
        if(dir.y > 0 || (dir.y == 0 && dir.x > 0)) {
            return ((int)lower.size() - 1 + Find(upper, dir)) % n;
        }
        else {
            return Find(lower, dir);
        }
    }
    point get(int l, int r, point p1, point p2) {
        int sl = sign(cross(p2 - p1, ps[l % n] - p1));
        while(l + 1 < r) {
            int mid = (l + r) >> 1;
            if(sign(cross(p2 - p1, ps[mid % n] - p1)) == sl) l = mid;
            else r = mid;
        }
        return getLL(p1, p2, ps[l % n], ps[(l + 1) % n]);
    }
    vector<point> getIS(point p1, point p2) {
        int X = findfarest((p2 - p1).turn90());
        int Y = findfarest((p1 - p2).turn90());
        if(X > Y) swap(X, Y);
        if(sign(cross(p2 - p1, ps[X] - p1)) * sign(cross(p2 - p1, ps[Y])) < 0) {
            return {get(X, Y, p1, p2), get(Y, X + n, p1, p2)};
        }
        else {
            return {};
        }
    }
    void update_tangent(point p, int id, int &a, int &b) {
        if(sign(cross(ps[a] - p, ps[id] - p)) > 0) a = id;
        if(sign(cross(ps[b] - p, ps[id] - p)) > 0) b = id;
    }
    void Binary_Search(int l, int r, point p, int &a, int &b) {
        if(l == r) return;
        update_tangent(p, l % n, a, b);
        int sl = sign(cross(ps[l % n] - p, ps[(l + 1) % n] - p));
        while(l + 1 < r) {
            int mid = (l + r) >> 1;
            if(sign(cross(ps[mid % n] - p, ps[(mid + 1) % n] - p)) == sl) l = mid;
            else r = mid;
        }
        update_tangent(p, r % n, a, b);
    }
    bool contain(point p) {//是否在凸包内
        if(p.x < lower[0].x || p.x > lower.back().x) return 0;
        int id = lower_bound(lower.begin(), lower.end(), point(p.x, -INF)) - lower.begin();
        if(lower[id].x == p.x) {
            if(lower[id].y > p.y) return 0;
        }
        else {
            if(sign(cross(lower[id] - lower[id - 1], p - lower[id - 1])) < 0) return 0;
        }
        id = lower_bound(upper.begin(), upper.end(), point(p.x, INF), greater<point>()) - upper.begin();
        if(upper[id].x == p.x) {
            if(upper[id].y < p.y) return 0;
        }
        else {
            if(sign(cross(upper[id] - upper[id - 2], p -  upper[id - 1])) < 0) return 0;
        }
        return 1;
    }
    bool get_tangent(point p, int &a, int &b) {//求切线
        if(contain(p)) return 0;
        a = b = 0;
        int id = lower_bound(lower.begin(), lower.end(), p) - lower.begin();
        Binary_Search(0, id, p, a, b);
        Binary_Search(id, lower.size(), p, a, b);
        id = lower_bound(upper.begin(), upper.end(), p, greater<point>()) - upper.begin();
        Binary_Search((int)lower.size() - 1, (int)lower.size() - 1 + id, p, a, b);
        Binary_Search((int)lower.size() - 1 + id, (int)lower.size() - 1 + upper.size(), p, a, b);
        return 0;
    }
};

```

### 三维旋转

```c++
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <set>
#include <vector>
#include <queue>
#include <cmath>
#include <cstring>
#include <ctime>
#include <map>
#define mp make_pair
#define pub push_back
#define pob pop_back
#define pof pop_front
#define pii pair<int, int>
#define fi first
#define se second
using namespace std;
const int N = 17;
const int M = 8000005;
const int INF = 1000000000;
typedef double db;
const db eps = 1e-8;
const db pi = acos(-1);
const db inf = 1e30;
inline int sign(db a) {return a < -eps ? -1 : a > eps;}
inline int cmp(db a, db b) {return sign(a - b);}
inline int inmid(db a, db b, db c) {return sign(a - c) * sign(b - c) <= 0;}
struct point {
    db x, y, z;
    point() {}
    point(db _x, db _y, db _z): x(_x), y(_y), z(_z) {}
    point operator + (const point &p) const {return point(x + p.x, y + p.y, z + p.z);}
    point operator - (const point &p) const {return point(x - p.x, y - p.y, z - p.z);}
    point operator * (db k) const {return point(x * k, y * k, z * k);}
    point operator / (db k) const {return point(x / k, y / k, z / k);}
    int operator == (const point &p) const {return !cmp(x, p.x) && !cmp(y, p.y) && !cmp(z, p.z);}
    db abs() {return sqrt(x * x + y * y + z * z);}
    db abs2() {return x * x + y * y + z * z;}
    db disto(point p) {return (*this - p).abs();}
    point unit() {db w = abs(); return point(x / w, y / w, z / w);}
    void scan() {db _x, _y, _z; scanf("%lf%lf%lf", &_x, &_y, &_z); x = _x, y = _y, z = _z;}
    void print() {printf("%.5f %.5f %.5f\n", x, y, z);}
}st, ed;
int inmid(point p1, point p2, point p3) {return inmid(p1.x, p2.x, p3.x) && inmid(p1.y, p2.y, p3.y);}
db dot(point p1, point p2) {return p1.x * p2.x + p1.y * p2.y + p1.z * p2.z;}
point cross(point p1, point p2) {return point(p1.y * p2.z - p1.z * p2.y, p1.z * p2.x - p1.x * p2.z, p1.x * p2.y - p1.y * p2.x);}
point proj(point p1, point p2, point q) {
    point p = p2 - p1;
    return p1 + p * (dot(q - p1, p) / p.abs2());
}
db disSP(point p1, point p2, point q) {
    point p3 = proj(p1, p2, q);
    return inmid(p1, p2, p3) ? q.disto(p3) : min(q.disto(p1), q.disto(p2));
}
point rotateP(point a, point b, double angle) {
    point e = cross(a, b);
    return a * cos(angle) + e * sin(angle);
}
int n, T; char s[5]; db d, turn, ans; point dir, up, le, nxt; const db loop = pi * 2;
int main() {
	scanf("%d", &T);
	while(T--) {
        st.scan(); ed.scan(); scanf("%d", &n); ans = inf;
        dir = point(1, 0, 0); up = point(0, 0, 1); le = point(0, 1, 0);
        while(n--) {
            scanf("%lf%s%lf", &d, s, &turn); nxt = st + dir * d;
            ans = min(ans, disSP(st, nxt, ed)); st = nxt;
            if(s[0] == 'U') {dir = rotateP(dir, le, turn); up = rotateP(up, le, turn);}
            else if(s[0] == 'D') {dir = rotateP(dir, le, -turn); up = rotateP(up, le, -			   turn);}
            else if(s[0] == 'L') {dir = rotateP(dir, up, -turn);le = rotateP(le, up, -             turn);}
            else {dir = rotateP(dir, up, turn); le = rotateP(le, up, turn);}
        }
        printf("%.2f\n", ans);
	}
	return 0;
}
```
### 辛普森积分
```c++
inline double f(double x) {//被积函数
    return 2.0 / (sqrt(sin(x) * sin(x) + 3) - sin(x));
}
inline double Simpson(double l, double r) {//辛普森法
    double m = (l + r) / 2;
    return (f(l) + 4 * f(m) + f(r)) * (r - l) / 6;
}
inline double asr(double l, double r, double res, double eps) {//递归自适应
    double m = (l + r) / 2;
    double L = Simpson(l, m), R = Simpson(m, r);
    if (fabs(L + R - res) <= 15 * eps) return L + R + (L + R - res) / 15;
    else return asr(l, m, L, eps / 2) + asr(m, r, R, eps / 2);
}
```
### 几何注意事项

$double$比较一定要$dcmp$

常数能提出来就提出来，尽量少做除法和乘法，必要时把浮点数扩大某个倍数

细心

根据不同题目可以修改点和直线的定义

注意舍入方式($0.5$的舍入方向) 防止输出$-0.$

几何题注意多测试不对称数据.

整数几何注意$xmult$和$dmult$是否会出界; 符点几何注意$eps$的使用.

避免使用斜率;注意除数是否会为$0$.

公式一定要化简后再代入.

判断同一个$2\pi$域内

两角度差应该是 $|a1-a2|<\beta\or|a1-a2|>2\pi-\beta$

相等应该是 $|a1-a2|<eps\or|a1-a2|>2\pi-eps$

需要的话尽量使用atan2 注意:$atan2(0, 0) = 0, atan2(1,0)=\pi/2,atan2(-1,0)=-\pi/2,atan2(0,1)=0,atan2(0,-1)=\pi$

$cross product = |u||v|sin(a)$

$dot product = |u||v|cos(a)$

$(P1-P0)\times(P2-P0)$结果的意义:

正: $<P0,P1>$在$<P0,P2>$顺时针$(0,\pi)$内

负: $<P0,P1>$在$<P0,P2>$逆时针$(0,\pi)$内

0 : $<P0,P1>, <P0,P2>$共线,夹角为$0$或$\pi$

10. 误差限缺省使用$1e-8$!

二.几何公式

三角形:

1. 半周长 $P=(a+b+c)/2$

2. 面积 $S=aHa/2=absin(C)/2=sqrt(P(P-a)(P-b)(P-c))$

3. 中线 $Ma=sqrt(2(b^2+c^2)-a^2)/2=sqrt(b^2+c^2+2bccos(A))/2$

4. 角平分线 $Ta=sqrt(bc((b+c)^2-a^2))/(b+c)=2bccos(A/2)/(b+c)$

5. 高线 $Ha=bsin(C)=csin(B)=sqrt(b^2-((a^2+b^2-c^2)/(2a))^2)$

6. 内切圆半径 $r=S/P=asin(B/2)sin(C/2)/sin((B+C)/2)=4Rsin(A/2)sin(B/2)sin(C/2)=sqrt((P-a)(P-b)(P-c)/P)=Ptan(A/2)tan(B/2)tan(C/2)$

7. 外接圆半径 $R=abc/(4S)=a/(2sin(A))=b/(2sin(B))=c/(2sin(C))$

四边形:

$D_1,D_1$为对角线,$M$对角线中点连线,$A$为对角线夹角

1. $a^2+b^2+c^2+d^2=D1^2+D2^2+4M^2$

2. $S=D_1D_2sin(A)/2$

(以下对圆的内接四边形)

3. $ac+bd=D_1D_2$

4. $S=sqrt((P-a)(P-b)(P-c)(P-d)),P$为半周长

正$n$边形:

$R$为外接圆半径, $r$为内切圆半径

1. 中心角 $A=2\pi/n$

2. 内角 $C=(n-2)\pi/n$

3. 边长 $a=2sqrt(R^2-r^2)=2Rsin(A/2)=2rtan(A/2)$

4. 面积 $S=nar/2=nr^2tan(A/2)=nR^2sin(A)/2=na^2/(4tan(A/2))$

圆:

1. 弧长$l=rA$

2. 弦长 $a=2sqrt(2hr-h^2)=2rsin(A/2)$

3. 弓形高 $h=r-sqrt(r^2-a^2/4)=r(1-cos(A/2))=atan(A/4)/2$

4. 扇形面积 $S1=rl/2=r^2A/2$

5. 弓形面积 $S2=(rl-a(r-h))/2=r^2(A-sin(A))/2$

棱柱:

1. 体积 $V=Ah,A$为底面积$,h$为高

2. 侧面积 $S=lp,l$为棱长$,p$为直截面周长

3. 全面积 $T=S+2A$

棱锥:

1. 体积$ V=Ah/3,A$为底面积$,h$为高

(以下对正棱锥)

2. 侧面积$ S=lp/2,l$为斜高$,p$为底面周长

3. 全面积 $T=S+A$

棱台:

1. 体积 $V=(A1+A2+sqrt(A1A2))h/3,A1.A2$为上下底面积$,h$为高(以下为正棱台)

2. 侧面积 $S=(p1+p2)l/2,p1.p2$为上下底面周长$,l$为斜高

3. 全面积 $T=S+A1+A2$

圆柱:

1. 侧面积 $S=2PIrh$

2. 全面积 $T=2PIr(h+r)$

3. 体积 $V=PIr^2h$

圆锥:

1. 母线 $l=sqrt(h^2+r^2)$

2. 侧面积 $S=PIrl$

3. 全面积 $T=PIr(l+r)$

4. 体积 $V=PIr^2h/3$

圆台:

1. 母线$ l=sqrt(h^2+(r_1-r_2)^2)$

2. 侧面积 $S=PI(r_1+r_2)l$

3. 全面积 $T=PIr_1(l+r_1)+PIr_2(l+r_2)$

4. 体积 $V=PI(r_1^2+r_2^2+r_1r_2)h/3$

球:

1. 全面积 $T=4\pi r^2$

2. 体积$ V=4\pi r^3/3$

球台:

1. 侧面积 $S=2\pi rh$

2. 全面积 $T=PI(2rh+r_1^2+r_2^2)$

3. 体积 $V=\pi h(3(r_1^2+r_2^2)+h^2)/6$

皮克定理是指一个计算点阵中顶点在格点上的多边形面积公式，该公式可以表示为$2S=2a+b-2$，其中$a$表示多边形内部的点数，$b$表示多边形边界上的点数，$s$表示多边形的面积。
##数据结构
###点分治(POJ1741)
```C++
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>
#define rep(i, a, b) for(int i = a; i < b; i++)
using namespace std;
const int maxn = 20020;
int to[maxn], next[maxn], w[maxn], head[maxn];
int dep[maxn], fa[maxn], siz[maxn], p[maxn]; bool done[maxn];
vector<int> tmp;
int cnt = 1, ans = 0, size, rt, n, k;
int read() {
	int x = 0, f = 1; char ch = getchar();
	while(ch < '0' || ch > '9') {if(ch == '-') f = -1; ch = getchar();}
	while(ch >= '0' && ch <= '9') {x = x * 10 + ch - '0'; ch = getchar();}
	return f * x;
}
void insert(int u, int v, int c) {
	to[++cnt] = v; next[cnt] = head[u]; w[cnt] = c; head[u] = cnt;
	to[++cnt] = u; next[cnt] = head[v]; w[cnt] = c; head[v] = cnt;
}
void getdep(int u, int f, int d) {
	siz[u] = 1; dep[u] = d; tmp.push_back(d);
	for(int i = head[u], v = to[i]; i; i = next[i], v = to[i]) if(v != f && !done[v]) {
		getdep(v, u, d + w[i]); siz[u] += siz[v];
	}
}
void findrt(int u, int f) {
	siz[u] = 1; p[u] = 0;
	for(int i = head[u], v = to[i]; i; i = next[i], v = to[i]) if(v != f && !done[v]) {
		findrt(v, u); siz[u] += siz[v];
		p[u] = max(p[u], siz[v]);
	}
	p[u] = max(p[u], size - siz[u]);
	if(p[u] < p[rt]) rt = u;
}
int calc(int u, int init) {
	int tot = 0;
	tmp.clear(); getdep(u, 0, init);
	sort(tmp.begin(), tmp.end());
	int l = 0, r = tmp.size() - 1;
	while(l < r) {
		if(tmp[l] + tmp[r] <= k) tot += r - l, l++;
		else r--;
	}
	return tot;
}
void work(int u) {
	ans += calc(u, 0); done[u] = 1;
	for(int i = head[u], v = to[i]; i; i = next[i], v = to[i]) if(!done[v]) {
		ans -= calc(v, w[i]); rt = 0;
		p[0] = size = siz[v]; findrt(v, 0);
		work(rt);
	}
}
int main() {
	while(1) {
		memset(done, 0, sizeof(done));
		memset(head, 0, sizeof(head));
		cnt = 1, ans = 0;
		n = read(), k = read();
		if(n == 0 && k == 0) break;
		rep(i, 1, n) {
			int u = read(), v = read(), w = read();
			insert(u, v, w);
		}
		work(1);
		printf("%d\n", ans);
	}
	return 0;
}
```
###树链剖分
```C++
namespace Tree {

	#define reg(i, u, v) for(int i = head[u], v = e[i].to; i; i = e[i].next, v = e[i].to)

	struct edge {
		int to, next;
	}e[maxn << 1];

	int head[maxn], fa[maxn], dep[maxn], siz[maxn], son[maxn], dfn[maxn], top[maxn], cnt, idx;

	void init() {cnt = idx = 0; clr(head); clr(dfn); clr(siz);}

	void ins(int u, int v) {e[++cnt] = (edge){v, head[u]}; head[u] = cnt;}

	void dfs1(int u, int d, int f) {
		fa[u] = f; dep[u] = d; siz[u] = 1; son[u] = 0;
		reg(i, u, v) if(v != f) {
			dfs1(v, d + 1, u); siz[u] += siz[v];
			if(siz[v] > siz[son[u]]) son[u] = v;
		}
	}

	void dfs2(int u, int t) {
		top[u] = t; dfn[u] = ++idx;
		if(son[u]) dfs2(son[u], t);
		reg(i, u, v) if(!dfn[v]) dfs2(v, v);
	}

	int lca(int u, int v) {
		int f1 = top[u], f2 = top[v];
		while(f1 != f2) {
			if(dep[f1] < dep[f2]) swap(f1, f2), swap(u, v);
			u = fa[f1]; f1 = top[u];
		}
		if(dep[u] > dep[v]) swap(u, v);
		return u;
	}

	void modify(int u, int v, int opt, ll val) {
		int f1 = top[u], f2 = top[v];
		while(f1 != f2) {
			if(dep[f1] < dep[f2]) swap(f1, f2), swap(u, v);
			Seg :: modify(1, n, dfn[f1], dfn[u], 1, opt, val);
			u = fa[f1]; f1 = top[u];
		}
		if(dep[u] > dep[v]) swap(u, v);
		Seg :: modify(1, n, dfn[u], dfn[v], 1, opt, val);
	}

	int query(int u, int v) {
		int f1 = top[u], f2 = top[v], ans = 0;
		while(f1 != f2) {
			if(dep[f1] < dep[f2]) swap(f1, f2), swap(u, v);
			ans += Seg :: query(1, n, dfn[f1], dfn[u], 1);
			u = fa[f1]; f1 = top[u];
		}
		if(dep[u] > dep[v]) swap(u, v);
		ans += Seg :: query(1, n, dfn[u], dfn[v], 1);
		return ans;
	}

}
```

###zkw线段树
```C++
namespace zkw {

	#define lson (t << 1)
	#define rson (t << 1 | 1)

	int tr[maxn << 2], M;

	void pushup(int t) {
		tr[t] = max(tr[lson], tr[rson]);
	}

	void build(int n) {
		M = 1; for(; M < n; M <<= 1);
		for(int i = 1; i <= n; i++) tr[i + M] = num[i];
		for(int i = M; i >= 1; i--) pushup(i);
	}

	void modify(int n, int x) {
		tr[n += M] = x;
		for(n >>= 1; n; n >>= 1) pushup(n);
	}

	int query(int l, int r) {
		int ret = 0;
		l--; r++; l += M; r += M;
		for(; (l^r^1); l>>=1, r>>=1) {
			if(l&1^1) ret = max(ret, tr[l^1]);
			if(r&1) ret = max(ret, tr[r^1]);
		}
		return ret;
	}
	
}
```

###区间加、乘线段树
```C++
namespace Seg {
	
	#define lson (t << 1)
	#define rson (t << 1) | 1

	int tr[maxn << 2], mul[maxn << 2], add[maxn << 2], siz[maxn << 2];

	void init() {
		clr(tr); clr(mul); clr(add); clr(siz);
	}

	void pushdown(int t) {
		tr[lson] *= mul[t]; tr[rson] *= mul[t];
		tr[lson] += add[t] * siz[lson]; tr[rson] += add[t] * siz[rson];
		mul[lson] *= mul[t]; mul[rson] *= mul[t];
		add[lson] *= mul[t]; add[rson] *= mul[t];
		add[lson] += add[t]; add[rson] += add[t];
		add[t] = 0; mul[t] = 1;
	}

	void pushup(int t) {
		tr[t] = tr[lson] + tr[rson];
	}

	void build(int l, int r, int t) {
		tr[t] = add[t] = 0; mul[t] = 1; siz[t] = r - l + 1;
		if(l == r) return;
		int mid = (l + r) >> 1;
		build(l, mid, lson); build(mid + 1, r, rson);
	}

	void modify(int l, int r, int ql, int qr, int t, int opt, int val) {
		pushdown(t); if(ql > qr) return;
		if(l == ql && qr == r) {
			if(opt == 1) {tr[t] += siz[t] * val; add[t] += val;}
			if(opt == 2) {tr[t] *= val; mul[t] *= val; add[t] *= val;}
			return;
		}
		int mid = (l + r) >> 1;
		if(qr <= mid) modify(l, mid, ql, qr, lson, opt, val);
		else if(ql > mid) modify(mid + 1, r, ql, qr, rson, opt, val);
		else modify(l, mid, ql, mid, lson, opt, val), modify(mid + 1, r, mid + 1, qr, rson, opt, val);
		pushup(t);
	}

	int query(int l, int r, int ql, int qr, int t) {
		pushdown(t); if(ql > qr) return 0;
		if(l == ql && qr == r) return tr[t]; if(ql > qr) return 0;
		int mid = (l + r) >> 1;
		if(qr <= mid) return query(l, mid, ql, qr, lson);
		if(ql > mid) return query(mid + 1, r, ql, qr, rson);
		return query(l, mid, ql, mid, lson) + query(mid + 1, r, mid + 1, qr, rson);
	}

}
```

###主席树
```C++
void insert(int x, int &y, int l, int r, int num) {
	y = ++cnt; ls[y] = ls[x]; rs[y] = rs[x];
	sum[y] = sum[x] + num;
	if(l == r) return;
	int mid = (l + r) >> 1;
	if(num <= mid) insert(ls[x], ls[y], l, mid, num);
	else insert(rs[x], rs[y], mid + 1, r, num);
}

int query(int x, int y, int l, int r, int num) {
	if(l == r) return sum[y] - sum[x];
	int mid = (l + r) >> 1;
	if(num <= mid) return query(ls[x], ls[y], l, mid, num);
	else return query(rs[x], rs[y], mid + 1, r, num) + sum[ls[y]] - sum[ls[x]];
}
```

###左偏树
```C++
namespace leftist {

	struct node {
		int ls, rs, h, score;
	}tr[maxn << 1];

	int merge(int a, int b) {
		if(a == 0) return b;
		if(b == 0) return a;
		if(tr[a].score > tr[b].score) swap(a, b);
		tr[a].rs = merge(tr[a].rs, b);
		if(tr[tr[a].ls].h < tr[tr[a].rs].h) swap(tr[a].ls, tr[a].rs);
		tr[a].h = tr[tr[a].rs].h;
		return a;
	}	

}
```

###k-d tree
```C++
namespace kd {

	struct point {
		int num[sigma];
		bool operator< (const point &a) const {return num[D] < a.num[D];}
	}a[maxn];

	struct node {
		int ch[2], num[sigma], val[sigma][2];
	}tr[maxn];

	void pushup(int t, int x) {
		for(int i = 0; i < sigma; i++)
			tr[t].val[i][0] = min(tr[t].val[i][0], tr[x].val[i][0]);
		for(int i = 0; i < sigma; i++)
			tr[t].val[i][1] = max(tr[t].val[i][1], tr[x].val[i][1]);
	}

	int build(int l, int r, int dir) {
		int mid = (l + r) >> 1; D = dir; nth_element(a + l, a + mid, a + r + 1);
		for(int i = 0; i < sigma; i++) tr[mid].num[i] = tr[mid].val[i][0] = tr[mid].val[i][1] = a[mid].num[i];
		if(l < mid) tr[mid].ch[0] = build(l, mid - 1, (dir + 1) % sigma), pushup(mid, tr[mid].ch[0]);
		if(mid < r) tr[mid].ch[1] = build(mid + 1, r, (dir + 1) % sigma), pushup(mid, tr[mid].ch[1]);
		return mid;
	}

}
```
###CDQ分治
```cpp
#include <cstdio>
#include <cstring>
#include <algorithm>
#define rep(i, a, b) for(int i = a; i <= b; i++)
#define lowbit(x) (x & (-x))
using namespace std;
const int maxn = 100100;
struct Query {
	int x, y, z, num;
	friend bool operator== (Query a, Query b) {
		return (a.x == b.x && a.y == b.y && a.z == b.z);
	}
	friend bool operator< (Query a, Query b) {
		if(a.y == b.y && a.z == b.z) return a.x < b.x;
		if(a.y == b.y) return a.z < b.z;
		return a.y < b.y;
	}
}q[maxn], tq[maxn];
int tr[maxn << 1], id[maxn << 1], cnt[maxn], ans[maxn], idx = 0, tot = 0, n, m;
bool cmp(Query a, Query b) {
	if(a.x == b.x && a.y == b.y) return a.z < b.z;
	if(a.x == b.x) return a.y < b.y;
	return a.x < b.x;
}
int read() {
	int x = 0, f = 1; char ch = getchar();
	while(ch < '0' || ch > '9') {if(ch == '-') f = -1; ch = getchar();}
	while(ch >= '0' && ch <= '9') {x = x * 10 + ch - '0'; ch = getchar();}
	return f * x;
}
void modify(int x, int val) {
	for(; x <= m; x += lowbit(x)) {
		if(id[x] == idx) tr[x] += val;
		else id[x] = idx, tr[x] = val;
	}
}
int query(int x) {
	int ret = 0;
	for(; x; x -= lowbit(x))
		if(id[x] == idx) ret += tr[x];
	return ret;
}
void CDQ(int l, int r) {
	int mid = (l + r) >> 1, l1 = l, l2 = mid + 1;
	if(l == r) {
		cnt[q[mid].x] += q[mid].num - 1;
		return;
	}
	idx++;
	rep(i, l, r) {
		if(q[i].x <= mid) modify(q[i].z, q[i].num);
		else cnt[q[i].x] += query(q[i].z);
	}
	rep(i, l, r) {
		if(q[i].x <= mid) tq[l1++] = q[i];
		else tq[l2++] = q[i];
	}
	rep(i, l, r) q[i] = tq[i];
	CDQ(l, mid); CDQ(mid + 1, r);
}
int main() {
	n = read(), m = read();
	rep(i, 1, n) q[i].x = read(), q[i].y = read(), q[i].z = read(), q[i].num = 1;
	sort(q + 1, q + n + 1, cmp);
	rep(i, 1, n)
		if(q[i] == q[i - 1]) q[tot].num++;
		else q[++tot] = q[i];
	rep(i, 1, n) q[i].x = i;
	sort(q + 1, q + tot + 1);
	CDQ(1, tot);
	rep(i, 1, tot) ans[cnt[q[i].x]] += q[i].num;
	rep(i, 0, n - 1) printf("%d\n", ans[i]);
	return 0;
}
```
### Splay

```c++
int T, n, m, w[N], cnt, rt, sz[N], fa[N], c[N][2], delta, leave, sum;
void init() {
    n = getint(), m = getint();
    memset(w, 0, sizeof(w));
    memset(sz, 0, sizeof(sz));
    memset(fa, 0, sizeof(fa));
    memset(c, 0, sizeof(c));
    delta = leave = cnt = rt = 0;
}
void pushup(int x) {
    int ls = c[x][0], rs = c[x][1];
    sz[x] = sz[ls] + sz[rs] + 1;
}
void rot(int x, int &k) {
    int y = fa[x], z = fa[y], l = c[y][1] == x, r = l ^ 1;
    if(y != k) c[z][c[z][1] == y] = x;
    else k = x;
    fa[x] = z, fa[y] = x, fa[c[x][r]] = y;
    c[y][l] = c[x][r], c[x][r] = y;
    pushup(y); pushup(x);
}
void splay(int x, int &k) {
    int y, z;
    while(x != k) {
        y = fa[x], z = fa[y];
        if(y != k) {
            if((c[z][0] == y) ^ (c[y][0] == x)) rot(x, k);
            else rot(y, k);
        }
        rot(x, k);
    }
}
void ins(int k) {
    if(!rt) {rt = ++cnt; w[cnt] = k, sz[cnt] = 1; return;}
    int p = rt, z;
    while(p > 0) {
        z = p; sz[p]++;
        if(w[p] > k) p = c[p][0];
        else p = c[p][1];
    }
    if(w[z] > k) c[z][0] = ++cnt;
    else c[z][1] = ++cnt;
    w[cnt] = k, sz[cnt] = 1, fa[cnt] = z;
    splay(cnt, rt);
}
int del(int &x, int f) {
    if(!x) return 0;
    int k, ls = c[x][0], rs = c[x][1], l = c[f][1] == x;
    if(w[x] + delta < m) {
        k = del(rs, x) + sz[ls] + 1;
        sz[rs] = sz[x] - k;
        sz[ls] = sz[x] = w[ls] = w[x] = 0;
        x = rs, fa[x] = f, c[f][l] = x;
    }
    else {k = del(ls, x); sz[x] -= k;}
    return k;
}
int findpos(int x, int k) {
    if(!x) return 0;
    int ls = c[x][0], rs = c[x][1];
    if(sz[ls] + 1 == k) return x;
    else if(sz[ls] >= k) return findpos(ls, k);
    return findpos(rs, k - sz[ls] - 1);
}
```

### LCT

```c++
struct LCT {
    int sz, c[2], fa, lazy;//此题询问到根的距离
}tree[N];
inline void pushup(int u) {
    tree[u].sz = tree[tree[u].c[0]].sz + tree[tree[u].c[1]].sz + 1;
}
inline void pushdown(int u) {
    if(tree[u].lazy) {
        tree[tree[u].c[0]].lazy ^= 1;
        tree[tree[u].c[1]].lazy ^= 1;
        swap(tree[u].c[0], tree[u].c[1]);
        tree[u].lazy ^= 1;
    }
}
inline bool ifroot(int u) {
    return tree[tree[u].fa].c[0] != u && tree[tree[u].fa].c[1] != u;
}
inline void rot(int x) {
    int y = tree[x].fa, z = tree[y].fa, p;
    p = tree[y].c[0] != x;
    if(!ifroot(y)) {
        if(tree[z].c[0] == y) tree[z].c[0] = x;
        else tree[z].c[1] = x;
    }
    tree[x].fa = z; tree[y].fa = x;
    tree[tree[x].c[p ^ 1]].fa = y;
    tree[y].c[p] = tree[x].c[p ^ 1];
    tree[x].c[p ^ 1] = y;
    pushup(y); pushup(x);
}
inline void downlazy(int u) {
    if(!ifroot(u)) downlazy(tree[u].fa);
    pushdown(u);
}
inline void splay(int x) {
    downlazy(x);
    while(!ifroot(x)) {
        int y = tree[x].fa, z = tree[y].fa;
        if(!ifroot(y)) {
            if((tree[y].c[0] == x) ^ (tree[z].c[0] == y)) rot(x);
            else rot(y);
        }
        rot(x);
    }
}
inline int access(int u) {
    int v = 0;
    while(u) {
        splay(u); tree[u].c[1] = v;
        pushup(u); v = u;
        u = tree[u].fa;
    }
    return v;
}
inline int findroot(int u) {
    access(u);
    splay(u);
    while(tree[u].c[0]) u = tree[u].c[0];
    return u;
}
inline void makeroot(int u) {
    access(u);
    splay(u);
    tree[u].lazy ^= 1;
}
inline void link(int u, int v) {
    makeroot(u);
    tree[u].fa = v;
}
inline void cut(int u) {
    access(u);
    splay(u);
    tree[u].c[0] = tree[tree[u].c[0]].fa = 0;
}
inline void cut(int u, int v) {
    /*access(u);
    splay(u);
    if(tree[u])
    tree[u].c[0] = tree[tree[u].c[0]].fa = 0;*/
    makeroot(u);
    access(v);
    splay(v);
    tree[u].fa = tree[v].c[0] = 0;
}
inline int ask(int u, int v) {
    makeroot(u);
    access(v);
    splay(v);
    return tree[v].sz;
}

```

### 笛卡尔树

```c++
top = 0;
memset(ls, 0, (n + 1) * 4);
memset(rs, 0, (n + 1) * 4);
memset(flag, 0, (n + 1));
for(int i = 1; i <= n; ++i) {
    int k = top;
  	while(k > 0 && a[stk[k]] < a[i]) --k;
    if(k) rs[stk[k]] = i;
    if(k < top) ls[i] = stk[k + 1];
    stk[++k] = i; top = k;
}
for(int i = 1; i <= n; ++i) flag[ls[i]] = flag[rs[i]] = 1;
for(int i = 1; i <= n; ++i) if(!flag[i]) {
    rt = i; break;
}

```

### 树状数组套主席树

```c++
int n, m, num, a[N], w[N << 1], wl, rt[N << 8], ls[N << 8], rs[N << 8], sum[N << 8], st[100], ed[100], stn, edn;
char s[5];
struct node {
    int l, r, k, type;
}op[N];
void Insert(int &root, int l, int r, int pre, int x, int v) {
    root = ++num;
    sum[root] = sum[pre] + v, ls[root] = ls[pre], rs[root] = rs[pre];
    if(l == r) return;
    int mid = (l + r) >> 1;
    if(x <= mid) Insert(ls[root], l, mid, ls[pre], x, v);
    else Insert(rs[root], mid + 1, r, rs[pre], x, v);
}
void add(int th, int val) {
    int k = lower_bound(w + 1, w + wl + 1, a[th]) - w;
    for(int i = th; i <= n; i += i & -i) Insert(rt[i], 1, wl, rt[i], k, val);
}
int query(int l, int r, int th) {
    if(l == r) return l;
    int res = 0, mid = (l + r) >> 1;
    for(int i = 1; i <= stn; ++i) res -= sum[ls[st[i]]];
    for(int i = 1; i <= edn; ++i) res += sum[ls[ed[i]]];
    if(th <= res) {
        for(int i = 1; i <= stn; ++i) st[i] = ls[st[i]];
        for(int i = 1; i <= edn; ++i) ed[i] = ls[ed[i]];
        return query(l, mid, th);
    }
    else {
        for(int i = 1; i <= stn; ++i) st[i] = rs[st[i]];
        for(int i = 1; i <= edn; ++i) ed[i] = rs[ed[i]];
        return query(mid + 1, r, th - res);
    }
}
int main() {
    scanf("%d%d", &n, &m);
    for(int i = 1; i <= n; ++i) {
        scanf("%d", &a[i]);
        w[++wl] = a[i];
    }
    for(int i = 1; i <= m; ++i) {
        scanf("%s", s);
        op[i].type = s[0] == 'C';
        scanf("%d%d", &op[i].l, &op[i].r);
        if(s[0] == 'C') {
            w[++wl] = op[i].r;
        }
        else {
            scanf("%d", &op[i].k);
        }
    }
    sort(w + 1, w + wl + 1);
    wl = unique(w + 1, w + wl + 1) - (w + 1);
    for(int i = 1; i <= n; ++i) add(i, 1);
    for(int i = 1; i <= m; ++i) {
        if(op[i].type) {
            add(op[i].l, -1);
            a[op[i].l] = op[i].r;
            add(op[i].l, 1);
        }
        else {
            stn = edn = 0;
            for(int j = op[i].l - 1; j; j -= j & -j) st[++stn] = rt[j];
            for(int j = op[i].r; j; j -= j & -j) ed[++edn] = rt[j];
            printf("%d\n", w[query(1, wl, op[i].k)]);
        }
    }
	return 0;
}

```

###一些结论
####求区间内有多少个不同的数
法一：主席树，每个点加入时不加权值，加入当前位置，并把上一个相同值得位置去掉，查询区间内大于某个值的数的个数即可
法二：莫队
法三：树状数组，算每个值在那一段中有贡献
####CDQ分治
将需要处理的操作分成两部分
计算左半部分的修改对右半部分的查询的贡献
递归处理左半部分操作
递归处理右半部分操作
##字符串
###kmp+AC自动机
```C++
void makefail(char *t, int p[], int len) {
    int i, j;
    p[i = 0] = j = -1;
    while(i < len) {
        if(j == -1 || t[i] == t[j]) {
            i++, j++;
            if(t[i] == t[j]) {
                p[i] = p[j];
            }
            else p[i] = j;
        }
        else j = p[j];
    }//nextval
    p[0] = p[1] = 0;
    for(int i = 1, j = 0; i <= len; ++i) {
        while(j > 0 && t[i] != t[j]) j = p[j];
        p[i + 1] = t[i] == t[j] ? ++j : 0;
    }//next
}
int c[N][26], tot, v[N], dis[N], fail[N], que[N];
void Insert(char *str) {
    int now = 0;
    for(int i = 0; str[i]; ++i) {
        int x = str[i] - 'a';
        if(!c[now][x]) c[now][x] = ++tot;
        now = c[now][x];
    }
    v[now] = 1;
}
void make_fail() {
    int he = 0, ta = 0;
    que[++ta] = 0;
    while(he < ta) {
        int x = que[++he];
        v[x] |= v[fail[x]];
        for(int i = 0; i < 26; ++i) {
            if(c[x][i]) {
                fail[c[x][i]] = x ? c[fail[x]][i] : 0;
                que[++ta] = c[x][i];
            }
            else c[x][i] = x ? c[fail[x]][i] : 0;
        }
    }
}
```

###SAM+求rank/height数组
```C++
namespace sam {

	#define rep(i, a, b) for(int i = a; i < b; i++)
	#define per(i, a, b) for(int i = (a - 1); i >= b; i--)

	struct node {
		int pa, len, ch[sigma]; bool tail;
	}tr[maxn];

	int last = 1, rt = 1, cnt = 1, tot = 0, n, id[maxn], pos[maxn], son[maxn][sigma];

	int extend(int x) {
		int p = last, np = ++cnt; tr[np].len = tr[p].len + 1; tr[np].tail = 1;
		id[tr[np].len] = np;
		for(; p && !tr[p].ch[x]; p = tr[p].pa) tr[p].ch[x] = np;
		if(!p) tr[np].pa = rt;
		else {
			int q = tr[p].ch[x];
			if(tr[p].len + 1 == tr[q].len) tr[np].pa = q;
			else {
				int nq = ++cnt; tr[nq] = tr[q]; tr[nq].tail = 0;
				tr[nq].len = tr[p].len + 1; tr[np].pa = tr[q].pa = nq;
				for(; p && tr[p].ch[x] == q; p = tr[p].pa) tr[p].ch[x] = nq;
			}
		}
		return last = np;
	}

	void dfs(int u) {
		if(tr[u].tail) {
			sa[rk[n - tr[u].len + 1] = ++tot] = n - tr[u].len + 1;
		}
		rep(i, 0, sigma) if(son[u][i]) dfs(son[u][i]);
	}


	void build_sa(char *ch) {
		n = strlen(ch);
		per(i, n, 0) extend(ch[i] - 'a');
		for(int i = n; i > 0; i--) {
			for(int x = id[i], p = n + 1; x && !pos[x]; x = tr[x].pa) {
				p -= tr[x].len - tr[tr[x].pa].len; pos[x] = p;
			}
		}
		for(int i = 2; i <= cnt; i++) {
			son[tr[i].pa][ch[pos[i] - 1] - 'a'] = i;
		}
		dfs(1);
		for(int i = 1; i <= n; i++) printf("%d ", sa[i]);
		puts("");


		int k = 0;

		for(int i = 1; i <= n; i++) {
			if(k) k--; int j = sa[rk[i] - 1];
			while(ch[i + k - 1] == ch[j + k - 1]) k++;
			height[rk[i]] = k;
		}

		for(int i = 2; i <= n; i++) printf("%d ", height[i]);
		puts("");

	}

}
```

###找一个最长字符串不包括任何禁止串
```C++
#include <cstdio>  
#include <cstring>  
#include <string>  
#include <algorithm>  
#include <map>  
#include <queue>  
using namespace std;  
const int MAXNODE = 50005;  
int n;   
struct AutoMac {  
    int ch[MAXNODE][26];  
    int val[MAXNODE];  
    int next[MAXNODE];  
    int sz;  
    void init() {  
    	sz = 1;   
    	memset(ch[0], 0, sizeof(ch[0]));  
    } 
    int idx(char c) {  
    	return c - 'A';  
    }  
    void insert(char *str, int v = 1) {  
    	int n = strlen(str);  
    	int u = 0;  
    	for (int i = 0; i < n; i++) {  
        	int c = idx(str[i]);  
        	if (!ch[u][c]) {  
        		memset(ch[sz], 0, sizeof(ch[sz]));  
        		val[sz] = 0;  
        		ch[u][c] = sz++;  
       	 	}  
        	u = ch[u][c];  
    	}  
    	val[u] = v;  
    }  
    void getnext() {  
    	queue<int> Q;  
    	next[0] = 0;      
    	for (int c = 0; c < n; c++) {  
        	int u = ch[0][c];  
        	if (u) {next[u] = 0; Q.push(u);}  
    	} 
    	while (!Q.empty()) {  
       		int r = Q.front(); Q.pop();  
        	for (int c = 0; c < n; c++) {  
        	int u = ch[r][c];  
        	if (!u) {  
            	ch[r][c] = ch[next[r]][c];  
            	continue;  
        	}  
        	Q.push(u);  
        	int v = next[r];  
        	while (v && !ch[v][c]) v = next[v];  
        	next[u] = ch[v][c];  
        	val[u] |= val[next[u]];  
    	}  
    }    
} gao;  
  
int t, m, vis[MAXNODE], dp[MAXNODE], zh[MAXNODE][2], vv[MAXNODE];  
char str[55];  
bool find(int u) {  
    vv[u] = 1;  
    for (int i = 0; i < n; i++) {  
    	int v = gao.ch[u][i];  
    	if (vis[v]) return true;  
   		if (!vv[v] && !gao.val[v]) {  
        	vis[v] = 1;  
        	if (find(v)) return true;  
        	vis[v] = 0;  
    	}  
    }  
    return false;  
}  
  
int dfs(int u) {  
    if (vis[u]) return dp[u];  
    vis[u] = 1;  
    dp[u] = 0;  
    for (int i = n - 1; i >= 0; i--) {  
    	if (!gao.val[gao.ch[u][i]]) {  
        	int tmp = dfs(gao.ch[u][i]) + 1;  
        	if (dp[u] < tmp) {  
        		dp[u] = tmp;  
        		zh[u][0] = gao.ch[u][i];  
        		zh[u][1] = i;  
        	}  
    	}  
    }  
    return dp[u];  
}  
void print(int u) {  
    if (zh[u][0] == -1) return;  
    printf("%c", zh[u][1] + 'A');  
    print(zh[u][0]);  
}  
  
int main() {  
    scanf("%d", &t);  
    while (t--) {  
    	gao.init();  
    	scanf("%d%d", &n, &m);  
    	while (m--) {  
        	scanf("%s", str);  
        	gao.insert(str);  
    	}  
    	gao.getnext();  
    	memset(vv, 0, sizeof(vv));  
    	memset(vis, 0, sizeof(vis));  
    	vis[0] = 1;  
    	if (find(0)) printf("No\n");  
    	else {  
        	memset(vis, 0, sizeof(vis));  
        	memset(zh, -1, sizeof(zh));  
        	if (dfs(0) == 0) printf("No\n");  
        	else {  
        		print(0);  
        		printf("\n");  
        	}  
    	}  
    }  
    return 0;  
} 
```
### 回文树

```c++
const int MAXN = 100005;
const int N = 26;
struct Palindromic_Tree {
    int next[MAXN][N];//next指针，next指针和字典树类似，指向的串为当前串两端加上同一个字符构成
    int fail[MAXN];//fail指针，失配后跳转到fail指针指向的节点
    int cnt[MAXN]; //表示节点i表示的本质不同的串的个数（建树时求出的不是完全的，最后count()函数跑一遍以后才是正确的）
    int num[MAXN]; //表示以节点i表示的最长回文串的最右端点为回文串结尾的回文串个数
    int len[MAXN];//len[i]表示节点i表示的回文串的长度（一个节点表示一个回文串）
    int S[MAXN];//存放添加的字符
    int last;//指向新添加一个字母后所形成的最长回文串表示的节点。
    int n;//表示添加的字符个数。
    int p;//表示添加的节点个数。
    int newnode(int l) {//新建节点
        for (int i = 0; i < N; ++i) next[p][i] = 0;
        cnt[p] = 0; num[p] = 0; len[p] = l;
        return p++;
    }
    void init() {//初始化
        p = 0; last = 0; n = 0;
        newnode(0); newnode(-1);
        S[n] = -1;//开头放一个字符集中没有的字符，减少特判
        fail[0] = 1;
    }
    int get_fail(int x) {//和KMP一样，失配后找一个尽量最长的
        while(S[n - len[x] - 1] != S[n]) x = fail[x];
        return x;
    }
    void add(int c) {
        c -= 'a';
        S[++n] = c;
        int cur = get_fail(last);//通过上一个回文串找这个回文串的匹配位置
        if(!next[cur][c]) {//如果这个回文串没有出现过，说明出现了一个新的本质不同的回文串
            int now = newnode(len[cur] + 2);//新建节点
            fail[now] = next[get_fail(fail[cur])][c] ;//和AC自动机一样建立fail指针，以便失配后跳转
            next[cur][c] = now; num[now] = num[fail[now]] + 1;
        }
        last = next[cur][c]; cnt[last]++;
    }
    void calc() {
        for(int i = p - 1; i >= 0; --i) cnt[fail[i]] += cnt[i];
        //父亲累加儿子的cnt，因为如果fail[v]=u，则u一定是v的子回文串！
    }
};

```

### manachar

```c++
void manacher(char str[], int h[]) {
    int n = strlen(str), m = 0;
    static char buf[M];
    for(int i = 0; i < n; ++i)
        buf[m++] = str[i], buf[m++] = '#';
    buf [m] = '\0';
    for(int i = 0, mx = 0, id = 0; i < m; ++i) {
        h[i] = i < mx ? min(mx - i, h[(id << 1) - i]) : 0;
        while(h[i] <= i && buf[i - h[i]] == buf[i + h[i]]) ++h[i];
        if(mx < i + h[i]) mx = i + h[i], id = i;
    }
}
```

### 最小表示法

```c++
int MinRep (int *s, int l) {
    int i = 0, j = 1, k;
    while(i < l && j < l) {
        for(k = 0; k < l && s[i + k] == s[j + k]; ++k);
        if(k == l) return i;
        if(s[i + k] > s[j + k]) {
            if(i + k + 1 > j) i = i + k + 1;
            else i = j + 1;
        }
        else if(j + k + 1 > i) j = j + k + 1;
        else j = i + 1;
    }
    return i < l ? i : j;
}
```

##模拟
###表达式求值（自定义运算优先级）
```C++
namespace Calculator {
	
	char str[1010]; stack<char> op; stack<pii> num;

	int priority(char opt) {
		if(opt == '(') return 0;
		if(opt == '+' || opt == '-') return 1;
		if(opt == '*' || opt == '/') return 2;
	}

	void calc() {
		int x, y, z; char opt;
		if(num.size() >= 2 && !op.empty()) {
			y = num.top(); num.pop();
			x = num.top(); num.pop(); 
			char opt = op.top(); op.pop();
			if(opt == '+') z = x + y;
			if(opt == '-') z = x - y;
			if(opt == '*') z = x * y;
			if(opt == '/') z = x / y;
			num.push(z);
		}
	}

	void work() {
		scanf("%s", str);
		int n = strlen(str);
		for(int i = 0; i < n; i++) {
			if(isdigit(str[i])) {
				int x = 0; while(isdigit(str[i])) {
					x = x * 10 + str[i++] - '0';
				}
				num.push(x);
			}

			char opt = str[i];
			
			if(i >= n) break;
			
			if(opt == '(') op.push(opt);
			else if(opt == ')') {
				while(!op.empty()) {
					if(op.top() == '(') {op.pop(); break;}
					calc();
				}
			} else {
				while(!op.empty() && priority(opt) <= priority(op.top())) calc();
				op.push(opt);
			}
		}
		while(!op.empty()) calc();
		printf("%d\n", num.top());
		while(!num.empty()) num.pop();
	}
}
```
###高精度
####C++实现（包括+ - * 单精度/)
```C++
#include <bits/stdc++.h>

#define rep(i, a, b) for(int i = a; i < b; ++i)
#define per(i, a, b) for(int i = (a - 1); i >= b; --i)
#define pb push_back

using namespace std;

typedef long long ll;

const int BASE = 10000, WIDTH = 4;

struct BigInteger {

	vector<int> s;

	void standardize() {
		per(i, s.size(), 0) if(s[i] == 0) s.pop_back(); else break;
		if(s.empty()) s.pb(0);
	}
 
	BigInteger& operator = (ll num) {
		s.clear(); for(; num; num /= BASE) s.pb(num % BASE);
		if(s.empty()) s.pb(0);
		return *this;
	}

	BigInteger& operator = (const string& num) {
		s.clear(); int len=(num.size() - 1) / WIDTH + 1, x = 0;
		rep(i, 0, len) {
			int ed = num.size() - i * WIDTH, st = max(0, ed - WIDTH);
			sscanf(num.substr(st, ed - st).c_str(), "%d", &x); s.pb(x);
		}
		standardize();
		return *this;
	}

	BigInteger operator + (const BigInteger& rhs) const {
		int siz=max(s.size(),rhs.s.size()), carry=0;
		BigInteger ans;
		rep(i, 0, siz) {
			int sum = carry;
			if(i < s.size()) sum += s[i];
			if(i < rhs.s.size()) sum += rhs.s[i];
			carry = sum / BASE;
			ans.s.pb(sum % BASE);
		}
		if(carry) ans.s.pb(carry);
		return ans;
	}

	BigInteger operator * (const BigInteger& rhs) const {
		BigInteger ans;
		rep(i, 0, rhs.s.size()) {
			BigInteger lans; int carry=0;
			rep(k, 0, i) lans.s.pb(0);
			rep(j, 0, s.size()) {
				int res = rhs.s[i] * s[j] + carry;
				carry = res / BASE; lans.s.pb(res % BASE);
			}
			while(carry) {
				lans.s.pb(carry % BASE);
				carry /= BASE;
			}
			ans = ans + lans;
		}
		return ans;
	}

	BigInteger operator - (const BigInteger& rhs) const {
		BigInteger ans; int carry=0;
		rep(i, 0, s.size()) {
			int diff = s[i] - carry;
			if(i < rhs.s.size()) diff -= rhs.s[i];
			carry = 0; while(diff < 0) ++carry, diff += BASE;
			ans.s.pb(diff);
		}
		ans.standardize();
		return ans;
	}

	BigInteger operator / (int rhs) const {
		BigInteger ans; vector<int> t; ll rmder = 0;
		per(i, s.size(), 0) {
			ll temp = rmder * BASE + s[i], div = temp / rhs;
			rmder = temp % rhs; t.pb(div);
		}
		per(i, t.size(), 0) ans.s.pb(t[i]);
		ans.standardize();
		return ans;
	}
 
	friend ostream& operator << (ostream& out,const BigInteger& rhs) {
		out << rhs.s.back();
		for(int i=rhs.s.size()-2;i>=0;--i) {
			char buf[5];
			sprintf(buf, "%04d", rhs.s[i]);
			cout << string(buf);
		}
		return out;
	}

	bool operator < (const BigInteger& rhs) const
	{
		if(s.size() != rhs.s.size()) return s.size()<rhs.s.size();
		per(i, s.size(), 0) if(s[i] != rhs.s[i]) return s[i] < rhs.s[i];
		return 0;
	}

};
```
####Java实现(BigInteger类)
```Java
BigInteger A, B;
A = new BigInteger("12345");
B = BigInteger.valueOf(37);
A = BigInteger.ONE; // ZERO or TEN are available
if(A.compareTo(B) < 0) {}
if(A.equals(B)) {}
int x = A.intValue();    // value should be in limit of int x
long y = A.longValue();  // value should be in limit of long y
String z = A.toString();
A = A.add(B);
A = A.subtract(B);
A = A.multiply(B);
A = A.divide(B);
A = A.mod(B);
BigInteger[] C = A.divideAndRemainder(B); // C[0] = A / B, C[1] = A % B;
A = A.shiftLeft(2);
A = A.shiftRight(2);
```
### java高精度开根

```java
import java.math.BigInteger;
import java.util.Scanner;
import java.util.TreeSet;
public class Main {
    static BigInteger n, mod;
    public static BigInteger Sqrt(BigInteger c) {
        if(c.compareTo(BigInteger.ONE)<=0) return c;//0返回0 1返回1
        BigInteger temp = null, x;
        x = c.shiftRight((c.bitLength() + 1) / 2);//初始猜值为二进制右移至位数只剩一半
        while(true) {//以下为牛顿迭代法
            temp = x; x = x.add(c.divide(x)).shiftRight(1);
            if(temp.equals(x)||x.add(BigInteger.ONE).equals(temp)) break;//两次迭代相等或只差一
        }
        return x;
    }
    public static boolean judge(BigInteger c) {
        BigInteger x = Sqrt(c);
        if(x.multiply(x).equals(c)) return true;//平方回去是否还是本身
        else return false;            
    }
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int t = sc.nextInt();
        while(t > 0) {
            t--; n = sc.nextBigInteger();
            boolean x = judge(n);
            boolean y = judge(n.multiply(n.subtract(BigInteger.ONE)).shiftRight(1));
            if(x && y) System.out.println("Arena of Valor");
            else if(!x && y) System.out.println("Clash Royale");
            else if(x && !y) System.out.println("Hearth Stone");
            else System.out.println("League of Legends");
        }
    }
}
```
### bitset

b.any() b中是否存在置为1的二进制位？ 

b.none() b中不存在置为1的二进制位吗？ 

b.count() b中置为1的二进制位的个数 

b.size() b中二进制位数的个数 

b[pos] 访问b中在pos处二进制位 

b.test(pos) b中在pos处的二进制位置为1么？ 

b.set() 把b中所有二进制位都置为1 

b.set(pos) 把b中在pos处的二进制位置为1

b.reset() 把b中所有二进制位都置为0

b.reset(pos) 把b中在pos处的二进制位置置为0

b.flip() 把b中所有二进制位逐位取反 

b.flip(pos) 把b中在pos处的二进制位取反 

b.to_ulong() 把b中同样的二进制位返回一个unsigned
##快速读入
###C++
```C++
#pragma GCC optimize(2)
namespace Quick_in {
	const int LEN=(1<<21)+1; char ibuf[LEN],*iH,*iT;int f,c;
	#define gc() (iH==iT?(iT=(iH=ibuf)+fread(ibuf,1,LEN,stdin),(iH==iT?EOF:*iH++)):*iH++)
	inline char nc(){
		while((c=gc())<=32)if(c==-1)return -1;
		return (char)c;
	}
	template<class T> inline void scan(T&x) {
		for (f=1,c=gc();c<'0'||c>'9';c=gc()) if (c=='-') f=-1;
		for (x=0;c<='9'&&c>='0';c=gc()) x=x*10+(c&15); x*=f;
	}
	template<class T> inline bool read(T&x) {
		for (f=1,c=gc();c<'0'||c>'9';c=gc()){ if(c==-1)return 0;if(c=='-') f=-1; }
		for(x=c-48;;x=x*10+(c&15)){ if(!isdigit(c=gc()))break;}x*=f; return 1;
	}
	inline int gline(char*s) {
		int l=-1;
		for (c=gc();c<=32;c=gc())if(c==-1)return c;
		for (;c>32||c==' ';c=gc()){
			s[++l]=c;
		}
		s[l+1]=0;
		return l;
	}
	inline int gs(char *s) {
		int l=-1;
		for (c=gc();c<=32;c=gc())if(c==-1)return c;
		for (;c>32;c=gc()){
			s[++l]=c;
		}
		s[l+1]=0;
		return l;
	}
	template <typename A, typename B> inline void scan(A&x,B&y){scan(x),scan(y);};
	template <typename A, typename B> inline bool read(A&x,B&y){return read(x)&&read(y);};
}
using Quick_in :: scan;
```
###Java
```Java
static class InputReader {
        private BufferedReader reader;
        private StringTokenizer tokenizer;

        public InputReader(InputStream stream) {
            reader = new BufferedReader(new InputStreamReader(stream), 32768);
            tokenizer = null;
        }

        public String next() {
            while (tokenizer == null || !tokenizer.hasMoreTokens()) {
                try {
                    tokenizer = new StringTokenizer(reader.readLine());
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
            return tokenizer.nextToken();
        }

        public int nextInt() {
            return Integer.parseInt(next());
        }

        public long nextLong() {
            return Long.parseLong(next());
        }
        
        public double nextDouble() {
            return Double.parseDouble(next());
        }
        
        public char[] nextCharArray() {
            return next().toCharArray();
        }
        
        public BigInteger nextBigInteger() {
            return new BigInteger(next());
        }
        
        public BigDecimal nextBigDecimal() {
            return new BigDecimal(next());
        }
        
    }
```
##对拍bash命令
```bash
#!/bin/bash
g++ $1.cpp -o $1 -O2 -std=c++11
g++ $2.cpp -o $2 -O2 -std=c++11
g++ $3.cpp -o gen -O2 -std=c++11
i=1;
while [ $i -le $4 ];
do
	./gen >tmp$i.in
	./$1 <tmp$i.in >tmp$i.out
	./$2 <tmp$i.in >tmp$i.ans
	if diff -b >tmp$i.diff tmp$i.out tmp$i.ans; then
		echo -e "Case #${i}:\t Accepted"
	else 
		echo -e "Case #${i}:\t Wrong Answer"
		break
	fi
	rm tmp$i.in tmp$i.out
	rm tmp$i.diff tmp$i.ans
let i=i+1
done
rm $1 $2 gen

# args[0] = Program A
# args[1] = Program B
# args[2] = Generator
# args[3] = Number of test cases
```
