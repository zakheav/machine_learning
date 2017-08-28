package word2vec;

import util.*;

import java.text.DecimalFormat;
import java.util.List;
import java.util.Map;

/**
 * Created by yuanxi.wy on 2017/8/25.
 */
public class Train {

    private static double likehood(double[] X, int predictWordIdx, HuffumanTree ht) {
        double likehood = 0.0;
        List<Integer> code = ht.treeArr[predictWordIdx].code;
        int nodeIdx = predictWordIdx;
        for (int i = 0; i < code.size(); ++i) {
            int d = code.get(i);
            nodeIdx = ht.parentArr[nodeIdx];
            double[] theta = ht.treeArr[nodeIdx].theta;
            likehood += d * Math.log(sigmoid(Vector.dotProduct(X, theta))) + (1 - d) * Math.log(1 - sigmoid(Vector.dotProduct(X, theta)));
        }
        return likehood;
    }

    private static double sigmoid(double a) {
        return 1.0 / (1.0 + Math.exp(-a));
    }

    public static void train(HuffumanTree hTree, Map<String, Integer> word_idx) {
        String[] sentence = GenerateDict.getSentence();

        while (sentence != null) {
            if (sentence.length < 1) continue;
            double[] deltaX = new double[Constants.VEC_SIZE];
            for (int i = 0; i < sentence.length; ++i) {
                int predictWordIdx = word_idx.get(sentence[i]);// 待预测的单词在huffuman树中的下标
                double[] X = new double[Constants.VEC_SIZE];// 待预测单词的上下文词向量的叠加
                for (int j = 0; j < sentence.length; ++j) {
                    if (i != j) {
                        X = Vector.add(X, hTree.treeArr[word_idx.get(sentence[j])].vec);
                    }
                }

                // debug
                System.out.println(sentence[i] + ": " + likehood(X, predictWordIdx, hTree));

                // out -> hidden 梯度上升
                List<Integer> code = hTree.treeArr[predictWordIdx].code;
                int nodeIdx = predictWordIdx;
                for (int j = 0; j < code.size(); ++j) {
                    int d = code.get(j);
                    nodeIdx = hTree.parentArr[nodeIdx];
                    double[] theta = hTree.treeArr[nodeIdx].theta;
                    double g = Constants.ALPHA * (d - sigmoid(Vector.dotProduct(X, theta)));// 梯度的一部分

                    // 更新待预测单词在huffuman树中路径上的每个非叶子节点的theta向量
                    hTree.treeArr[nodeIdx].theta = Vector.add(theta, Vector.scalarMulti(g, X));
                    // 更新X向量
                    deltaX = Vector.add(deltaX, Vector.scalarMulti(g, theta));
                }

            }
            // hidden -> input 梯度上升
            for (int i = 0; i < sentence.length; ++i) {
                hTree.treeArr[word_idx.get(sentence[i])].vec = Vector.add(deltaX, hTree.treeArr[word_idx.get(sentence[i])].vec);
            }
            sentence = GenerateDict.getSentence();
        }
    }
}
