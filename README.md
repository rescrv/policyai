# PolicyAI

Unstructured data is everywhere.  PolicyAI is a framework for turning unstructured data into
structured data via unstructured policies.  Given a structure for data (a type declaration) the
PolicyAI framework makes it possible to write policies that transform an unstructured input into a
the typed, structured output.

It is indeed a refinement of structured outputs such that:
- Composable statements yield compatible outputs.
- Conflicting statements are detected.

For example, consider a simple policy about what should happen with an email.  We basically know
that we want to mark the email read/unread, categorize and label it, prioritize human interaction
with it, and reply with a drafted response using a template.  What we can say is that the email
should default to unread, but be toggle-able.  The priority must take the highest priority from any
policy.  The category must be uniformly agreed upon, but labels are an open set of strings.
Finally, the template, if present, must be agreed to by all present.  Such a policy has a type like:

```text
type policyai::EmailPolicy {
    unread: bool = true,
    priority: ["low", "medium", "high"] @ highest wins,
    category: ["ai", "distributed systems", "other"] @ agreement = "other",
    template: string @ agreement,
    labels: [string],
}
```

Conceptually speaking, policies couple unstructured _semantic injections_ that describe when the
policy applies and what to do with structured actions.  Consider a sample email policy for how to
handle email messages.  A set of policies about email could take this form:

```text
Policy #0:
The email is relevant to football of either form.
Mark "unread" false with "low" "priority".
{
    "unread": false,
    "priority": "low"
}

Policy #1:
The email pertains to ecommerce.
Add "Shopping" to "labels".
{
    "labels": [
      "Shopping"
    ]
}

Policy #2:
The email is from mom@example.org.
Record "high" "priority" and add "Family" to "labels".
{
    "priority": "high",
    "labels": [
      "Family"
    ]
  }
}
```

There's plenty of room for cross-policy interaction here.  What should the outcome of applying these
policies be if my mom sends me an email about shopping for football-related gear?  It should be
marked as unread with high priority and the labels "Family" and "Shopping".  That's the unit-tested
outcome for these policies and this query.

Notice how the policies are compositional.  Each policy specifies a fragment of what to do with the
data; sometimes policies compose and sometimes they conflict.  PolicyAI is capable of detecting
conflicts between policies (but currently does not report them).

For further illustration of the problems that come from composing policies, consider a manager for
GitHub issues and notifications.  Giving an open source model instructions about an issue's relative
priority based upon the issue's assignee yields poor results because the model will often cross
instructions from different assignees.  For example, I asked a model to prioritize messages from
Alice and deprioritize messages from Bob in the most straightforward way possible and what it
decided was that no decision could be reached because of conflicting information because Alice must
be priority and Bob must not be priority.

I want the reliability of o1 with the speed, cost, and license of phi4.

PolicyAI is about hitting that mark.

## What is a policy?

To define the term precisely, a policy is a semantic injection coupled with a structured output.
Policies are designed in such a way that policies of the same type compose a larger policy in a
reliable way.

A set of policies are said to be compatible when:
- They have the same policy type.
- They do not specify conflicting output.

The first property is a static property.  Given a set of policies, they either have the same type or
they don't.  The second property is a runtime property.  If no actions conflict, the property is
trivially satisfied, and this can be verified statically, but if different actions assign different
values to the same output, the conflict can only be detected at runtime when an LLM tries to apply
both.
