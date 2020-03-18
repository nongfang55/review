# coding=gbk
import dulwich.diff_tree
import dulwich.objects
import dulwich.patch
from dulwich.repo import Repo


class dulwichDemo:
    """duwich demo的尝试"""

    @staticmethod
    def demo():
        repo = Repo(r"C:\Users\ThinkPad\Desktop\Python\rails")
        # print(repo)
        # print(repo.get_description())
        commit = repo.get_object("fba1064153d8e2f4654df7762a7d3664b93e9fc8".encode("ascii"))
        print(commit.tree)
        # print(type(commit))
        # print(commit.author)
        # print(commit.committer)
        # print(commit.message)
        # print(commit.tree)
        tree = dulwichDemo.getObject(repo, commit.tree)
        # for item in tree.items():
        #     print(item)
        #     obj = (dulwichDemo.getObject(repo, item.sha))
        #     print(obj)
        #     if isinstance(obj, dulwich.objects.Blob):
        #         print(obj.splitlines().__len__())
        #         for line in obj.splitlines():
        #             print(line)
        commit2 = repo.get_object("8cdef19142792101b24e1124daa434b8171bf0f2")
        # print(commit2)
        # print(commit2.tree)
        diff = dulwich.diff_tree.tree_changes(repo.object_store, commit2.tree, commit.tree, include_trees=True,
                                              change_type_same=False)
        # print(diff)
        for d in diff:
            print(d.old)
            print(dulwichDemo.getObject(repo, d.old.sha))
            print(dulwichDemo.getObject(repo, d.new.sha))
            with open("diff{0}.txt".format(d.old.sha), "wb+") as f:
                dulwich.patch.write_blob_diff(f, d.old, d.new)
                print("-"*50)
        # with open("diff.txt", "wb+") as f:
        #     dulwich.patch.write_tree_diff(f, repo.object_store, commit2.tree, commit.tree)

    @staticmethod
    def getObject(repo, sha):
        return repo.get_object(sha)


if __name__ == "__main__":
    dulwichDemo.demo()
