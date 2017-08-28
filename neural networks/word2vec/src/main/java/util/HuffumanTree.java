package util;

import java.util.*;

/**
 * Created by yuanxi.wy on 2017/8/21.
 */
class WordComparator implements Comparator<Word> {
    public int compare(Word w0, Word w1) {
        return w0.counter == w1.counter ? 0 : ((w0.counter - w1.counter) > 0 ? -1 : 1);
    }
}

public class HuffumanTree {
    public Word[] treeArr;// 里面包含了叶子结点和非叶子节点（非叶子结点数为n，叶子节点数为n+1）
    public int[] parentArr;// 记录每个节点的父节点下标（根节点的父节点下标是0）

    public HuffumanTree(List<Word> Dict) {// 传入的是叶子结点
        treeArr = new Word[Dict.size() * 2 - 1];
        parentArr = new int[Dict.size() * 2 - 1];
        int[] left_right = new int[Dict.size() * 2 - 1];// 记录每个节点是父节点的左枝0还是右枝1，小的在左边，大的在右边
        Collections.sort(Dict, new WordComparator());// 把传入的字典按照word中的counter从大到小排序

        for (int i = 0; i < treeArr.length; ++i) {
            if (i < Dict.size()) {
                treeArr[i] = Dict.get(i);
            } else {
                treeArr[i] = new Word();
                treeArr[i].counter = Long.MAX_VALUE;
            }
        }
        // 构建huffuman树
        int ptr1 = Dict.size() - 1;// 这个指针永远在叶子节点的半区（左半区）
        int ptr2 = ptr1 + 1;// 这个节点永远在非叶子节点的半区（右半区）
        int newNodePtr = ptr2;
        int counter = Dict.size();
        while (counter < Dict.size() * 2 - 1) {// huffman树的节点还没有构造完
            int min1Idx = 0;// 当前最小的节点的下标
            int min2Idx = 0;// 当前第二小的节点的下标

            // 寻找当前最小的节点的下标
            if (ptr1 >= 0) {
                if (treeArr[ptr1].counter < treeArr[ptr2].counter) min1Idx = ptr1--;
                else min1Idx = ptr2++;
            } else min1Idx = ptr2++;

            // 寻找当前第二小的节点的下标
            if (ptr1 >= 0) {
                if (treeArr[ptr1].counter < treeArr[ptr2].counter) min2Idx = ptr1--;
                else min2Idx = ptr2++;
            } else min2Idx = ptr2++;

            parentArr[min1Idx] = newNodePtr;
            left_right[min1Idx] = 0;
            parentArr[min2Idx] = newNodePtr;
            left_right[min2Idx] = 1;
            treeArr[newNodePtr].counter = treeArr[min1Idx].counter + treeArr[min2Idx].counter;
            ++newNodePtr;
            ++counter;
        }

        // 根据huffuman树，获取每个单词在树中对应的编码
        for (int i = 0; i < Dict.size(); ++i) {
            Stack<Integer> stack = new Stack<Integer>();
            int j = i;
            List<Integer> code = new ArrayList<Integer>();
            while (parentArr[j] > 0) {// 自底向上获取编码
                code.add(left_right[j]);
                j = parentArr[j];
            }
            treeArr[i].code = code;
        }
    }

    public static void main(String[] args) {
        List<Word> dict = new ArrayList<Word>();
        for (int i = 1; i <= 7; ++i) {
            Word w = new Word(i + "");
            w.counter = i;
            dict.add(w);
        }
        HuffumanTree ht = new HuffumanTree(dict);
        for (int i = 0; i < dict.size(); ++i) {
            System.out.print(ht.treeArr[i].word + ": ");
            for (int c : ht.treeArr[i].code) {
                System.out.print(c + ",");
            }
            System.out.println();
        }
    }
}
