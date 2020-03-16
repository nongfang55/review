# coding=gbk
from ctypes import *

"""FPS 算法在c++当中使用到的class"""


class c_fps_review(Structure):
    _fields_ = [
        ("repo_full_name", c_char_p),
        ("pull_number", c_int),
        ("submitted_at", c_char_p),
        ("commit_id", c_char_p),
        ("user_login", c_char_p)
    ]


class c_fps_pr(Structure):
    _fields_ = [
        ("repo_full_name", c_char_p),
        ("number", c_int),
        ("user_login", c_char_p)
    ]


class c_fps_commit(Structure):
    _fields_ = [
        ("sha", c_char_p),
    ]


class c_fps_file(Structure):
    _fields_ = [
        ("commit_sha", c_char_p),
        ("filename", c_char_p)
    ]


class c_fps_result(Structure):
    _fields_ = [
        ("answer", c_char * 500),
        ("recommend", c_char * 500)
    ]


class FPSClassCovert:
    """用来clas转换的帮助类"""

    @staticmethod
    def convertPullRequest(pullrequests):
        res = []
        for pr in pullrequests:
            c_pr = c_fps_pr()
            c_pr.repo_full_name = bytes(pr.repo_full_name, encoding='utf-8')
            c_pr.number = pr.number
            c_pr.user_login = bytes(pr.user_login, encoding='utf-8')
            res.append(c_pr)
        return res

    @staticmethod
    def convertReview(reviews):
        res = []
        for review in reviews:
            c_review = c_fps_review()
            c_review.repo_full_name = bytes(review.repo_full_name, encoding='utf-8')
            c_review.pull_number = review.pull_number
            c_review.commit_id = bytes(review.commit_id, encoding='utf-8')
            c_review.submitted_at = bytes(review.submitted_at, encoding='utf-8')
            c_review.user_login = bytes(review.user_login, encoding='utf-8')
            res.append(c_review)
        return res

    @staticmethod
    def convertCommit(commits):
        res = []
        for commit in commits:
            c_commit = c_fps_commit()
            c_commit.sha = bytes(commit.sha, encoding='utf-8')
            res.append(c_commit)
        return res

    @staticmethod
    def convertFile(files):
        res = []
        for file in files:
            c_file = c_fps_file()
            c_file.commit_sha = bytes(file.commit_sha, encoding='utf-8')
            c_file.filename = bytes(file.filename, encoding='utf-8')
            res.append(c_file)
        return res
