/* 主机端验证：用与 Python 相同的输入序列跑 C 场景分类器，输出 csv 供比对。 */
#include <stdio.h>
#include <stdlib.h>
#include "scene_classifier.h"

int main(void)
{
    SceneClassifier_Init();

    FILE* fin = fopen("scene_test_data.csv", "r");
    FILE* fout = fopen("scene_test_c.csv", "w");
    if (!fin || !fout) { fprintf(stderr, "cannot open io\n"); return 1; }

    char line[256];
    while (fgets(line, sizeof(line), fin)) {
        float ms, bmp, t;
        if (sscanf(line, "%f,%f,%f", &ms, &bmp, &t) != 3) continue;
        SceneClassifier_Update(ms, bmp, t);
        fprintf(fout, "%.8f,%.8f,%d\n",
                SceneClassifier_Prob[0], SceneClassifier_Prob[1],
                SceneClassifier_Pred);
    }
    fclose(fin);
    fclose(fout);
    printf("C inference done -> scene_test_c.csv\n");
    return 0;
}
