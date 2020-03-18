# coding=gbk
from source.utils.StringKeyUtils import StringKeyUtils
import json


class GraphqlHelper:
    """返回graphql 需要的query语句"""

    @staticmethod
    def getTimeLineQueryByNodes(body):
        """返回查询timeline需要的语句"""
        body[StringKeyUtils.STR_KEY_QUERY] = GraphqlHelper.STR_KEY_QUERY_PR_TIMELINE
        return body

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
              }
            
            ... on HeadRefForcePushedEvent {
              afterCommit {
                oid
              }
              beforeCommit {
                oid
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
