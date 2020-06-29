# coding=gbk
from source.utils.StringKeyUtils import StringKeyUtils
import json


class GraphqlHelper:
    """返回graphql 需要的query语句"""

    # @staticmethod
    # def getTimeLineQueryByNodes(body):
    #     """返回查询timeline需要的语句"""
    #     body[StringKeyUtils.STR_KEY_QUERY] = GraphqlHelper.STR_KEY_QUERY_PR_TIMELINE
    #     return body

    @staticmethod
    def getGraphlQuery(body, query):
        """body 加入query语句"""
        if query is not None:
            body[StringKeyUtils.STR_KEY_QUERY] = query
        return body

    @staticmethod
    def getTimeLineQueryByNodes(body):
        """返回查询timeline需要的语句"""
        body[StringKeyUtils.STR_KEY_QUERY] = GraphqlHelper.STR_KEY_QUERY_PR_TIMELINE
        return body
    @staticmethod
    def getPrInformationByNumber():
        return GraphqlHelper.STR_KEY_QUERY_PR_ALL

    @staticmethod
    def getGraphqlVariables(body, args=None):
        """返回传递参数需要的json"""
        if args is None or isinstance(body, dict) is False:
            body[StringKeyUtils.STR_KEY_VARIABLES] = GraphqlHelper.STR_KEY_NONE
        else:
            body[StringKeyUtils.STR_KEY_VARIABLES] = json.dumps(args)
        return body

    # @staticmethod
    # def getGraphqlArg(args=None):
    #     """返回传递参数需要的json"""
    #     pass

    STR_KEY_QUERY_VIEWER = "{viewer{name}}"

    STR_KEY_NONE = "{}"

    STR_KEY_QUERY_PR_TIMELINE = '''
query($ids:[ID!]!) { 
  nodes(ids:$ids) {
    ... on PullRequest {
      id
      author {
        login
      }
      timelineItems(first:100) {
        edges {
          node {
            __typename
            ... on Node {
               id
            }
              
            ... on PullRequestCommit {
              commit {
                oid
              }
            }
            
            ... on PullRequestReview {
              commit {
                oid
              }
              author {
                login
              }
              comments(first: 100) {
                nodes {
                  commit {
                    oid
                  }
                  originalCommit {
                    oid
                  }
                  author {
                    login
                  }
                  path
                }
              }
            }
            ... on HeadRefForcePushedEvent {
              afterCommit {
                oid
              }
              beforeCommit {
                oid
              }
            }
            ... on PullRequestReviewThread {
              id
              comments(first: 100) {
                nodes {
                  commit {
                    oid
                  }
                  originalCommit {
                    oid
                  }
                  author {
                    login
                  }
                  path
                }
              }
            }
            ... on MergedEvent {
              id
              commit {
                oid
              }
            }
            ... on IssueComment {
              author {
                login
              }
            }
          }   
        }
        }
      }
    }
  rateLimit {
    limit
    cost
    remaining
    resetAt
  }
}
    '''

    STR_KEY_QUERY_PR_ALL = '''query($name:String!, $owner:String!, $number:Int!) { 
      
      viewer {
       login
      }
	  
	  
      rateLimit {
        limit
        cost
        remaining
        resetAt
      }
	  
	  
      repository(name:$name, owner:$owner) { 
          issueOrPullRequest(number:$number) { 
           __typename
			 
			 
           ... on Issue {
              number
            }
			
			
           ... on PullRequest {
             # pull request的信息  23
             # 少 repo_full_name, comments
             # 少 review_comments, commits
             # 少 head, base
             
             number 
             databaseId
             id
             # id 是node id
             state
             title
             author {
               login
             }
             body
             createdAt
             updatedAt
             closedAt
             mergedAt
             mergeCommit {
               oid
             }
             authorAssociation
             merged
             additions
             deletions
             changedFiles
			 
			 
             #issue comment
             comments(first:50) {
              nodes {
               # 9项 少 repo_full_name,pull_number
                databaseId
                id
                author {
                  login
                }
                createdAt
                updatedAt
                authorAssociation
                body
              }
            }
			
			
            # review
            reviews(first:50) {
              nodes{
               # 11项 少 repo_full_name,pull_number,user_login
               databaseId
               author{
                 login
               }
               body
               state
               authorAssociation
               submittedAt
               commit {
                 oid
               }
               id
			  
			  # review comment 内嵌
              comments(first:50) {
                nodes {
                 # 21项 少pull_request_review_id,
                 # startline,orignal_start_line,start_side,line,origin_line
                 # side
                 databaseId
                 author {
                   login
                 }
                 body
                 diffHunk
                 path
                 commit {
                   oid
                 }
                 position
                 originalPosition
                 originalCommit{
                   oid
                 }
                 createdAt
                 updatedAt
                 authorAssociation
                 replyTo {
                   databaseId
                 }
                 id
                }
              }
			  
			  
              # review 涉及的commit
              commit {
                # 15个属性 少status_total,commit_comment_count
                # commit_author_date, commit_committer_date
                oid
                id
                author {
                name
                email
                }
                committer {
                 name
                 email
                }
                messageBody
                additions
                deletions
                changedFiles
                parents(first:50){
                  nodes {
                   oid
                  }
                 }
                }
			  
			  
              }
            }
			
			
            # user 
            participants(first:50){
            nodes {
               #共计26项 少 type,followers_url,
               #following_url, starred_url, subscriptions_url,
               #organizations_url, repos_url, events_url
               #received_events_url, blog, public_repos,
               #public_gists, followers, following
               login
               isSiteAdmin
               databaseId
               email
               id
               name
               company
               location
               isHireable
               bio
               createdAt
               updatedAt
             }
            }
			
			
            #files
            files(first:50) {
               nodes {
               # 新增项 pr直接关联的文件变化
               path
               additions
               deletions
               }
            }
			
			
            # pr 直接相关的commit
            commits(first:50) {
             nodes{
              commit {
              # 15个属性 少status_total,commit_comment_count
              # commit_author_date, commit_committer_date
               oid
               id
               author {
                name
                email
               }
               committer {
                 name
                 email
                }
                messageBody
                additions
                deletions
                changedFiles
                parents(first:50){
                  nodes {
                   oid
                  }
                }
              }
             }
            }
			
			
            # head branch
            headRef {
             name
             prefix
             id
             repository {
               name
               nameWithOwner
             }
            }
            headRefOid
            headRefName
            headRepository {
              name
              nameWithOwner
            }
			
			
            # base branch
            baseRef {
             name
             prefix
             id
             repository {
               name
               nameWithOwner
             }
            }
            baseRefOid
            baseRefName
            baseRepository {
              name
              nameWithOwner
            }
			
			
           }
        }  
      } 
    }'''